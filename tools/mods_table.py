"""Gộp mọi kết quả MODS (tools/eval_mods.py) thành một bảng CSV.

    python tools/mods_table.py                      # results/mods/*.json → docs/data/mods_results.csv

Mỗi dòng là một `<out>.json` (bỏ `_frames.json`). Recall theo kích thước (toàn khung) đọc từ bảng
"Detections based on sizes" trong `<out>_stats.log` nếu đã chạy `--stats`. Dòng `_int8` là mask do
Hailo-8L trên Pi ghi; còn lại là FP32 trên laptop (PyTorch hoặc ONNX).
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = ["tiny", "very_small", "small", "medium", "large", "very_large"]


def size_recall(log):
    """Dòng số đầu tiên sau tiêu đề "Detections based on sizes" (6 giá trị %), hoặc None."""
    if not log.exists():
        return None
    lines = log.read_text().splitlines()
    for i, line in enumerate(lines):
        if "Detections based on sizes" in line:
            for row in lines[i + 1:i + 8]:
                vals = re.findall(r"\d+(?:\.\d+)?|nan", row)
                if len(vals) == 6 and "tiny" not in row:
                    return vals
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/mods"))
    ap.add_argument("--out", default=str(ROOT / "docs/data/mods_results.csv"))
    args = ap.parse_args()

    head = ["result", "precision", "f1", "pr", "re", "tp", "fp", "fn", "f1_dz", "pr_dz", "re_dz", "tp_dz", "fp_dz",
            "fn_dz", "we_rmse_px", *[f"recall_{s}" for s in SIZES]]
    rows = [",".join(head)]
    for f in sorted(Path(args.results).glob("*.json")):
        if f.stem.endswith("_frames"):
            continue
        d = json.loads(f.read_text())
        o, z = d["obstacles"], d["danger_zone"]
        sr = size_recall(f.with_name(f.stem + "_stats.log")) or [""] * 6
        rows.append(",".join(map(str, [
            f.stem, "int8" if f.stem.endswith("_int8") else "fp32",
            round(100 * o["f1"], 2), round(100 * o["precision"], 2), round(100 * o["recall"], 2), o["tp"], o["fp"], o["fn"],
            round(100 * z["f1"], 2), round(100 * z["precision"], 2), round(100 * z["recall"], 2), z["tp"], z["fp"], z["fn"],
            d["water_edge"]["rmse_px"], *sr])))
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"{len(rows) - 1} dòng → {args.out}")


if __name__ == "__main__":
    main()
