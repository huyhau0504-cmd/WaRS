"""Chấm mask dự đoán trên LaRS val (Žust và cs., ICCV 2023) — giao thức nội bộ, không phải toolkit LaRS chính thức.

    python tools/predict.py --onnx exports/e7_ewasr_h_lite_384x512.onnx --images ~/datasets/lars/val/images --out preds/lars/e7
    python tools/eval_lars.py --pred-masks preds/lars/e7/masks --lars-root ~/datasets/lars --scene river_like

Nhãn semantic của LaRS: 0 vật cản, 1 nước, 2 trời, 255 bỏ qua (cùng quy ước lớp với MaSTr1325).
- IoU theo lớp trên mọi pixel có nhãn (bỏ 255).
- Recall vật cản theo loại: mỗi vật (segment "thing" trong panoptic, diện tích ≥ --min-area) được tính là tìm
  thấy khi ≥ --coverage-thr pixel của nó bị dự đoán là vật cản (tiêu chí coverage như MODS).
- Báo động sai: tỉ lệ pixel nước (nhãn) bị dự đoán là vật cản, và số vùng vật cản dự đoán (≥ --min-area) nằm
  trọn trong vùng nước nhãn (không chạm pixel vật cản nhãn nào) trung bình mỗi ảnh.
"""
import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

NAMES = {0: "vật cản", 1: "nước", 2: "trời"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-masks", required=True, help="thư mục <stem>.png mã lớp 0/1/2 ở độ phân giải ảnh gốc")
    ap.add_argument("--lars-root", default=str(Path.home() / "datasets/lars"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--scene", default="all", choices=["all", "river_like", "sea_like"])
    ap.add_argument("--coverage-thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=25)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    root = Path(args.lars_root).expanduser() / args.split
    attrs = {Path(a["file_name"]).stem: a["labels"] for a in json.load(open(root / "image_annotations.json"))["annotations"]}
    pan = json.load(open(root / "panoptic_annotations.json"))
    cats = {c["id"]: c["name"] for c in pan["categories"]}
    stuff = {1, 3, 5}  # Static Obstacle, Water, Sky
    conf = np.zeros((3, 3), np.int64)
    found, total = collections.Counter(), collections.Counter()
    fp_regions, water_px, water_to_obst, n_img = 0, 0, 0, 0
    for ann in pan["annotations"]:
        stem = Path(ann["file_name"]).stem
        if args.scene != "all" and attrs[stem]["scene_type"] != args.scene:
            continue
        pred = cv2.imread(str(Path(args.pred_masks) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        sem = cv2.imread(str(root / "semantic_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise SystemExit(f"thiếu mask dự đoán {stem}")
        if pred.shape != sem.shape:
            pred = cv2.resize(pred, (sem.shape[1], sem.shape[0]), interpolation=cv2.INTER_NEAREST)
        n_img += 1
        valid = sem != 255
        conf += np.bincount(sem[valid].astype(np.int64) * 3 + pred[valid], minlength=9).reshape(3, 3)
        obst = pred == 0
        # vật cản theo loại
        rgb = cv2.imread(str(root / "panoptic_masks" / ann["file_name"]))[:, :, ::-1].astype(np.int64)
        pid = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        for s in ann["segments_info"]:
            if s["category_id"] in stuff or s["area"] < args.min_area:
                continue
            m = pid == s["id"]
            total[s["category_id"]] += 1
            found[s["category_id"]] += obst[m].mean() >= args.coverage_thr
        # báo động sai trên mặt nước
        water = sem == 1
        water_px += water.sum()
        water_to_obst += (obst & water).sum()
        gt_obst = cv2.dilate((sem == 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        n, lab, st, _ = cv2.connectedComponentsWithStats((obst & ~(sem == 2)).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if st[i, 4] >= args.min_area:
                comp = lab == i
                if not (comp & gt_obst).any() and water[comp].mean() > 0.9:
                    fp_regions += 1
    iou = {NAMES[c]: float(conf[c, c] / max(conf[c].sum() + conf[:, c].sum() - conf[c, c], 1)) for c in range(3)}
    res = {"split": args.split, "scene": args.scene, "images": n_img, "iou": iou, "miou": float(np.mean(list(iou.values()))),
           "recall_by_class": {cats[c]: f"{found[c]}/{total[c]}" for c in sorted(total)},
           "recall_things": float(sum(found.values()) / max(sum(total.values()), 1)),
           "water_to_obstacle_pct": float(100 * water_to_obst / max(water_px, 1)),
           "fp_regions_per_image": fp_regions / max(n_img, 1)}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
