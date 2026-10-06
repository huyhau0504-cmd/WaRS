"""Đánh giá dự đoán trên USV-VN theo giao thức trong wars/eval/usv_metrics.py.

    # Tiêu chí chính (coverage, kiểu MODS) — dùng chung cho phân đoạn và YOLO (box được tô thành mask)
    python tools/evaluate_usv.py --pred-masks preds/ewasr_fp32/masks --split test
    python tools/evaluate_usv.py --pred-boxes preds/yolov8n/boxes.json --split test
    # Tiêu chí phụ IoU ≥ 0.5 (chuẩn phát hiện đối tượng; bất lợi cho phân đoạn vì vật cản dính bờ)
    python tools/evaluate_usv.py --pred-boxes preds/yolov8n/boxes.json --split test --mode iou
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.data.usv_vn import label_path_for, list_images, load_water_edges, read_boxes  # noqa: E402
from wars.eval.usv_metrics import (Accumulator, boxes_to_mask, eval_boxes_iou, eval_mask_coverage,  # noqa: E402
                                   mask_to_boxes, water_region)
from wars.postprocess import load_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-masks", help="thư mục <stem>.png mã lớp")
    ap.add_argument("--pred-boxes", help="JSON {stem: [[x1,y1,x2,y2,score], ...]} theo pixel ảnh gốc")
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=["coverage", "iou"], default="coverage")
    ap.add_argument("--water-edges", default="data/usv_vn/water_edges.json")
    ap.add_argument("--coverage-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--score-thr", type=float, default=0.25, help="ngưỡng điểm cho --pred-boxes")
    ap.add_argument("--size-edges", type=float, nargs="+", default=[32, 96],
                    help="ngưỡng sqrt(diện tích box) chia nhóm kích thước")
    ap.add_argument("--out-json")
    args = ap.parse_args()
    if bool(args.pred_masks) == bool(args.pred_boxes):
        ap.error("cần đúng một trong --pred-masks / --pred-boxes")

    edges = load_water_edges(args.water_edges)
    boxes_pred = json.loads(Path(args.pred_boxes).read_text()) if args.pred_boxes else None
    acc = Accumulator(args.size_edges)
    missing_edges = 0
    for p in list_images(args.usv_root, args.split):
        h, w = cv2.imread(str(p)).shape[:2]
        gt = read_boxes(label_path_for(p), w, h)
        water = water_region(edges[p.stem], h, w) if p.stem in edges else None
        missing_edges += water is None

        if boxes_pred is not None:
            preds = [b for b in boxes_pred.get(p.stem, []) if b[4] >= args.score_thr]
            if args.mode == "iou":
                hits, fp = eval_boxes_iou(preds, gt, water, args.iou_thr)
            else:
                hits, fp = eval_mask_coverage(boxes_to_mask(preds, h, w), gt, water, args.coverage_thr,
                                              args.min_area)
        else:
            mask = load_mask(Path(args.pred_masks) / f"{p.stem}.png", (h, w))
            if args.mode == "coverage":
                hits, fp = eval_mask_coverage(mask, gt, water, args.coverage_thr, args.min_area)
            else:
                hits, fp = eval_boxes_iou(mask_to_boxes(mask, water, args.min_area), gt, water, args.iou_thr)
        acc.add(gt, hits, fp)

    res = {"split": args.split, "mode": args.mode, **acc.summary()}
    if missing_edges:
        print(f"[!] {missing_edges} ảnh chưa có mép nước trong {args.water_edges} → không tính FP/Precision/F1")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
