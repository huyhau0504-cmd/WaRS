"""So mask INT8 (Hailo emulator / Pi) với mask FP32 cùng mô hình: % pixel trùng, nhầm lẫn nước→vật cản,
nước→trời, và recall coverage USV-VN (tools/evaluate_usv.py).

    python tools/compare_masks.py --ref preds/e1_ewasr_fp32/masks --split test \\
        preds/emu/e1_ewasr_c64/masks preds/emu/e1_ewasr_c1024/masks
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def confusion(pred_dir, ref_dir):
    """Ma trận nhầm lẫn 3×3 (% pixel; hàng = tham chiếu, cột = dự đoán) và số ảnh."""
    cm, n = np.zeros((3, 3), np.int64), 0
    for p in sorted(Path(pred_dir).glob("*.png")):
        ref = cv2.imread(str(Path(ref_dir) / p.name), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        cm += np.bincount(ref.ravel().astype(np.int64) * 3 + m.ravel(), minlength=9).reshape(3, 3)
        n += 1
    return cm / max(cm.sum(), 1) * 100, n


def recall(pred_dir, split):
    out = subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_usv.py")), "--pred-masks",
                          str(pred_dir), "--split", split], capture_output=True, text=True).stdout
    try:
        r = json.loads(out[out.index("{"):out.rindex("}") + 1])
        return f"{r['tp']}/{r['gt']}"
    except (ValueError, KeyError):
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", nargs="+", help="thư mục mask cần so")
    ap.add_argument("--ref", required=True, help="thư mục mask tham chiếu (FP32)")
    ap.add_argument("--split", help="split USV-VN để chấm recall (bỏ trống = không chấm)")
    args = ap.parse_args()
    print("| thư mục | ảnh | trùng | nước→vật cản | nước→trời | recall |\n|---|---|---|---|---|---|")
    for d in args.pred:
        c, n = confusion(d, args.ref)
        rec = recall(d, args.split) if args.split else ""
        print(f"| {d} | {n} | {np.trace(c):.2f}% | {c[1, 0]:.2f}% | {c[1, 2]:.2f}% | {rec} |")


if __name__ == "__main__":
    main()
