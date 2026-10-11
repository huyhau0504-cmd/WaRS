"""Smoke test đầu-cuối trên CPU: train → resume → export ONNX → predict (PyTorch, ONNX) → evaluate.

    python tools/smoke_test.py                 # eWaSR-H, dưới 3 phút trên laptop
    python tools/smoke_test.py --models all    # mọi biến thể trong wars.models.MODELS
    python tools/smoke_test.py --keep          # giữ thư mục tạm để xem log từng bước

Dùng dữ liệu giả (wars/data/synthetic.py) ở 192×256. Bước predict dùng 4 ảnh đầu của
data/usv_vn/test nếu có (khi đó chạy thêm bước evaluate), không thì dùng ảnh giả.
In bảng PASS/FAIL/SKIP (SKIP = bước trước nó đã FAIL); thoát mã 1 nếu có bước FAIL.
"""
import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wars.data.synthetic import make_mastr_like  # noqa: E402
from wars.data.usv_vn import label_path_for, list_images  # noqa: E402
from wars.models import MODELS  # noqa: E402

SIZE = (192, 256)
N_PRED = 4


class StepFailed(Exception):
    pass


def run(cmd, log_path):
    """Chạy một script Python của repo; ghi stdout+stderr vào log. Lỗi → StepFailed."""
    p = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    Path(log_path).write_text(out)
    if p.returncode != 0:
        raise StepFailed(f"mã thoát {p.returncode}, xem {log_path}")
    return out


def check(cond, msg):
    if not cond:
        raise StepFailed(msg)


def metrics_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def smoke_model(name, tmp, data, pred_src, results):
    out = tmp / name
    out.mkdir()
    common = ["--config", "configs/base.yaml", "--out", out, "--set", f"model.name={name}",
              f"model.input_size=[{SIZE[0]},{SIZE[1]}]", "train.batch_size=2", "train.workers=0",
              "train.amp=false", "model.init_weights=null", *[f"data.{k}={v}" for k, v in data.items()]]
    onnx_path = out / f"{name}.onnx"
    preds = {"torch": out / "pred_torch", "onnx": out / "pred_onnx"}

    def train():
        run(["tools/train.py", *common, "train.epochs=2"], out / "train.log")
        for f in ("last.pt", "best.pth", "metrics.csv"):
            check((out / f).exists(), f"thiếu {f}")
        rows = metrics_rows(out / "metrics.csv")
        check(len(rows) == 2, f"metrics.csv có {len(rows)} dòng, cần 2")
        check(all(math.isfinite(float(r["loss"])) for r in rows), "loss không hữu hạn")
        return f"loss {float(rows[-1]['loss']):.4f}"

    def resume():
        log = run(["tools/train.py", *common, "train.epochs=3"], out / "resume.log")
        check("Tiếp tục từ epoch 2" in log, "log không có 'Tiếp tục từ epoch 2'")
        rows = metrics_rows(out / "metrics.csv")
        check(len(rows) == 3, f"metrics.csv có {len(rows)} dòng, cần 3")
        check((out / "final.pth").exists(), "thiếu final.pth")
        return "3 epoch"

    def export():
        log = run(["tools/export_onnx.py", "--model", name, "--weights", out / "best.pth",
                   "--size", *SIZE, "--out", onnx_path], out / "export.log")
        m = re.search(r"max\|torch-onnx\|=([0-9.eE+-]+)", log)
        check(m and onnx_path.exists(), "không thấy file ONNX hoặc dòng sai số")
        err = float(m.group(1))
        check(err < 1e-3, f"sai số torch–onnx {err:.2e} ≥ 1e-3")
        return f"sai số {err:.1e}"

    def predict():
        src = (["--usv-root", ROOT / "data/usv_vn", "--split", "test"] if pred_src["usv"]
               else ["--images", pred_src["dir"]]) + ["--limit", N_PRED]
        run(["tools/predict.py", "--model", name, "--weights", out / "best.pth", "--size", *SIZE,
             "--out", preds["torch"], *src], out / "predict_torch.log")
        run(["tools/predict.py", "--onnx", onnx_path, "--out", preds["onnx"], *src], out / "predict_onnx.log")
        agree = []
        for p in pred_src["images"]:
            h, w = cv2.imread(str(p)).shape[:2]
            m_t = cv2.imread(str(preds["torch"] / "masks" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
            m_o = cv2.imread(str(preds["onnx"] / "masks" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
            check(m_t is not None and m_o is not None, f"thiếu mask của {p.stem}")
            check(m_t.shape == (h, w) == m_o.shape, f"mask {p.stem} sai kích thước")
            agree.append((m_t == m_o).mean())
        check(min(agree) >= 0.99, f"PyTorch và ONNX chỉ trùng {min(agree):.2%} pixel")
        return f"{len(agree)} ảnh, trùng ≥ {min(agree):.2%}"

    def evaluate():
        # Thư mục USV-VN tạm chỉ chứa các ảnh đã predict (symlink, không sao chép)
        usv = out / "usv"
        for p in pred_src["images"]:
            for src, sub in ((p, "images"), (label_path_for(p), "labels")):
                dst = usv / "test" / sub / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    dst.symlink_to(src.resolve())
        res = out / "usv_test.json"
        run(["tools/evaluate_usv.py", "--pred-masks", preds["torch"] / "masks", "--usv-root", usv,
             "--split", "test", "--out-json", res], out / "evaluate.log")
        check(res.exists(), "không có file --out-json")
        return "đã ghi usv_test.json"

    steps = [("train", train), ("resume", resume), ("export", export), ("predict", predict)]
    if pred_src["usv"]:
        steps.append(("evaluate", evaluate))
    failed = False
    for step, fn in steps:
        if failed:
            results.append((name, step, "SKIP", 0.0, ""))
            continue
        t0 = time.time()
        try:
            detail, status = fn(), "PASS"
        except StepFailed as e:
            detail, status, failed = str(e), "FAIL", True
        results.append((name, step, status, time.time() - t0, detail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ewasr_h", help="tên biến thể, cách nhau bằng dấu phẩy, hoặc 'all'")
    ap.add_argument("--keep", action="store_true", help="giữ thư mục tạm")
    args = ap.parse_args()
    names = list(MODELS) if args.models == "all" else args.models.split(",")

    t_start = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="wars_smoke_"))
    try:
        data = make_mastr_like(tmp / "mastr", n=8, size=SIZE)
        usv_images = list_images(ROOT / "data/usv_vn", "test")[:N_PRED]
        pred_src = ({"usv": True, "images": usv_images} if len(usv_images) == N_PRED else
                    {"usv": False, "dir": tmp / "mastr/images",
                     "images": sorted((tmp / "mastr/images").glob("*.jpg"))[:N_PRED]})
        results = []
        for name in names:
            smoke_model(name, tmp, data, pred_src, results)
    finally:
        if args.keep:
            print(f"Giữ thư mục tạm: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'mô hình':<10} {'bước':<9} {'kết quả':<7} {'giây':>6}  chi tiết")
    for name, step, status, sec, detail in results:
        print(f"{name:<10} {step:<9} {status:<7} {sec:>6.1f}  {detail}")
    n_fail = sum(r[2] == "FAIL" for r in results)
    print(f"\nTổng thời gian {time.time() - t_start:.0f} s · ảnh predict: "
          f"{'USV-VN test' if pred_src['usv'] else 'dữ liệu giả'} · {'FAIL' if n_fail else 'PASS'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
