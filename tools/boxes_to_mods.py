"""Khung bao (tools/vlm_detect.py) → mask màu MODS để chấm bằng toolkit chính thức (tools/eval_mods.py).

    python tools/boxes_to_mods.py --boxes runs/vlm/owlv2/boxes.json --score-thr 0.2 --out preds/mods/owlv2_t020
    python tools/eval_mods.py --pred preds/mods/owlv2_t020 --mods-root ~/datasets/mods --sequences $(cat configs/mods_vlm_subset.txt) \
        --out-json results/mods_vlm/owlv2_t020.json

Pixel trong hộp (điểm ≥ --score-thr) = vật cản, còn lại = nước, giống cách tô hộp YOLO trên USV-VN. Toolkit chấm vật
cản theo độ phủ ≥ 50% và đếm báo động sai là vùng vật cản trên mặt nước (dưới mép nước nhãn), nên hộp dư ra phía bờ
không bị phạt. Sai số mép nước trong kết quả không có ý nghĩa với phương pháp chỉ trả hộp.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.postprocess import MODS_COLORS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True, help="thư mục <method>/<seq>/<frame>.png")
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--size", type=int, nargs=2, default=[958, 1278], help="H W ảnh MODS")
    ap.add_argument("--images-dir", help="LaRS/ảnh bất kỳ: lấy kích thước từ ảnh gốc <stem>.jpg và ghi mask mã lớp "
                                         "<out>/<stem>.png (0 vật cản, 1 nước) cho tools/eval_lars.py")
    ap.add_argument("--max-area-frac", type=float, default=1.0,
                    help="bỏ hộp phủ quá tỉ lệ này của ảnh (VLM đôi khi trả một hộp phủ cả khung)")
    ap.add_argument("--drop-labels", nargs="*", default=[],
                    help="bỏ hộp có nhãn chứa một trong các từ này (vd. water sky shore)")
    args = ap.parse_args()

    res = json.loads(Path(args.boxes).read_text())
    n_box = 0
    for key, boxes in res["frames"].items():
        if args.images_dir:
            H, W = cv2.imread(str(next(Path(args.images_dir).glob(f"{key}.*")))).shape[:2]
        else:
            H, W = args.size
        mask = np.ones((H, W), np.uint8)  # 1 = nước
        for x1, y1, x2, y2, score, _ in boxes:
            if score < args.score_thr or (x2 - x1) * (y2 - y1) > args.max_area_frac * H * W:
                continue
            if any(w in str(_).lower() for w in args.drop_labels):
                continue
            x1, x2 = sorted((int(round(max(0, min(W, x1)))), int(round(max(0, min(W, x2))))))
            y1, y2 = sorted((int(round(max(0, min(H, y1)))), int(round(max(0, min(H, y2))))))
            mask[y1:y2, x1:x2] = 0
            n_box += 1
        p = Path(args.out) / f"{key}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), mask if args.images_dir else cv2.cvtColor(MODS_COLORS[mask], cv2.COLOR_RGB2BGR))
    print(f"{len(res['frames'])} khung, {n_box} hộp (điểm ≥ {args.score_thr}) → {args.out}")


if __name__ == "__main__":
    main()
