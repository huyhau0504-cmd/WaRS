"""Chấm mask dự đoán trên MODS bằng toolkit chính thức (github.com/bborja/mods_evaluation, commit ghim).

    python tools/predict.py --model ewasr --weights weights/ewasr_resnet18.pth --mods ~/datasets/mods --out preds/mods/ewasr
    python tools/eval_mods.py --pred preds/mods/ewasr --mods-root ~/datasets/mods --out-json results/mods/ewasr.json
    python tools/eval_mods.py ... --sequences 1 2      # chỉ vài sequence (thử nhanh)

`--pred` là thư mục `<seq>/<frame>.png` (mask màu MODS_COLORS, như `tools/predict.py --mods` ghi); tên thư mục
là tên phương pháp. Toolkit được clone vào `third_party/mods_evaluation` (không commit) và vá bằng
`third_party/mods_evaluation.patch` (thêm tổng TP/FP/FN vào JSON kết quả). Toolkit chạy bằng venv riêng
`third_party/.venv-mods` (yacs, scikit-image, matplotlib, prettytable, colorama, tqdm, scipy, opencv) để không
động vào `.venv` chính.

Ngưỡng của toolkit (configs/config.py): MIN_OVERLAP 0,5 (độ phủ để tính TP), AREA_THRESHOLD 25 px (áp cho cả
vật cản GT và vùng dự đoán khi đếm FP), vùng nguy hiểm 15 m. Mask dự đoán được toolkit resize về 1278×958
bằng nearest, nên lưu ở kích thước đầu vào mô hình như WaSR là đủ.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.postprocess import MODS_COLORS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/bborja/mods_evaluation.git"
COMMIT = "b243918cb142d76dee72141a562db94d6aa7649c"  # 25/03/2026, "temporary dataset link change"


def ensure_toolkit(path, patch):
    if not (path / "modb_evaluation.py").exists():
        subprocess.run(["git", "clone", "-q", REPO, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"{path} đang ở {head}, cần {COMMIT}")
    applied = subprocess.run(["git", "-C", str(path), "diff", "--quiet"]).returncode != 0
    if not applied:
        subprocess.run(["git", "-C", str(path), "apply", str(patch)], check=True)


def run_stats(args, toolkit, tmp, method, out, summary):
    """get_evaluation_statistics.py → summary["statistics"] (theo kích thước/loại vật cản); log ở <out>_stats.log."""
    st = subprocess.run([args.python, "get_evaluation_statistics.py", method, "--config-file", str(tmp / "cfg.yaml")],
                        cwd=toolkit, capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"})
    out.with_name(out.stem + "_stats.log").write_text(st.stdout + st.stderr)
    sj = tmp / f"results_{method}_summary.json"
    if sj.exists():
        summary["statistics"] = json.loads(sj.read_text())


def prf(tp, fp, fn):
    pr = tp / (tp + fp) if tp + fp else 0.0
    re = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": pr, "recall": re, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="thư mục <method>/<seq>/<frame>.png")
    ap.add_argument("--mods-root", required=True)
    ap.add_argument("--sequences", type=int, nargs="+", help="chỉ số sequence 1..94 (mặc định: tất cả)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--toolkit", default=str(ROOT / "third_party/mods_evaluation"))
    ap.add_argument("--python", default=str(ROOT / "third_party/.venv-mods/bin/python"))
    ap.add_argument("--stats", action="store_true", help="chạy thêm get_evaluation_statistics.py (theo kích thước/loại)")
    ap.add_argument("--stats-only", action="store_true",
                    help="chỉ chạy thống kê trên <out-json>_frames.json đã có (không chấm lại)")
    args = ap.parse_args()

    toolkit, pred = Path(args.toolkit), Path(args.pred).resolve()
    mods = Path(args.mods_root).expanduser().resolve()
    ensure_toolkit(toolkit, ROOT / "third_party/mods_evaluation.patch")
    method = pred.name
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        tmp = Path(tmp)
        colors = "\n".join(f"    - {list(map(int, c))}" for c in MODS_COLORS)
        (tmp / "cfg.yaml").write_text(
            f"PATHS:\n  RESULTS: '{tmp}'\n  DATASET: '{mods}'\n  DATASET_CALIB: '{mods / 'calibration'}'\n"
            f"  SEGMENTATIONS: '{pred.parent}'\nSEGMENTATIONS:\n  SEQ_FIRST: False\n  INPUT_COLORS:\n{colors}\n"
            "VISUALIZATION:\n  SEQUENCE_PROGRESS: False\n")
        frames = out.with_name(out.stem + "_frames.json")
        if args.stats_only:
            summary = json.loads(out.read_text())
            (tmp / f"results_{method}.json").write_text(frames.read_text())
            run_stats(args, toolkit, tmp, method, out, summary)
            out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"{method}: thống kê → {out}" if "statistics" in summary else f"{method}: thống kê lỗi, xem log")
            return
        cmd = [args.python, "modb_evaluation.py", method, "--config-file", str(tmp / "cfg.yaml"),
               "--workers", str(args.workers)]
        if args.sequences:
            cmd += ["--sequences", *map(str, args.sequences)]
        t0 = time.time()
        log = subprocess.run(cmd, cwd=toolkit, capture_output=True, text=True)
        (out.with_suffix(".log")).write_text(log.stdout + log.stderr)
        if log.returncode:
            raise SystemExit(f"toolkit lỗi (xem {out.with_suffix('.log')}):\n{log.stderr[-2000:]}")
        res = json.loads((tmp / f"results_{method}.json").read_text())
        s = res["summary"]
        summary = {
            "method": method, "pred": str(pred), "mods_root": str(mods), "toolkit_commit": COMMIT,
            "sequences": args.sequences or "all", "eval_seconds": round(time.time() - t0, 1),
            "parameters": res["parameters"],
            "obstacles": {"tp": s["tp"], "fp": s["fp"], "fn": s["fn"], **prf(s["tp"], s["fp"], s["fn"])},
            "danger_zone": {"tp": s["tp_d"], "fp": s["fp_d"], "fn": s["fn_d"], **prf(s["tp_d"], s["fp_d"], s["fn_d"])},
            "water_edge": {"rmse_px": s["we_rmse_px"], "overshoot_pct": s["we_overshoot_pct"],
                           "undershoot_pct": s["we_undershoot_pct"],
                           "water_land_detection_pct": 100 * s["water_land_tp"] / max(s["water_land_tp"] + s["water_land_fn"], 1)},
        }
        if args.stats:
            run_stats(args, toolkit, tmp, method, out, summary)
        (tmp / f"results_{method}.json").replace(frames)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    o, d = summary["obstacles"], summary["danger_zone"]
    print(f"{method}: F1 {100 * o['f1']:.1f}% (Pr {100 * o['precision']:.1f}, Re {100 * o['recall']:.1f}) · "
          f"vùng nguy hiểm F1 {100 * d['f1']:.1f}% · mép nước RMSE {summary['water_edge']['rmse_px']:.0f} px → {out}")


if __name__ == "__main__":
    main()
