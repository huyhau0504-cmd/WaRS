"""Ghép mask kiểu MaSTr1325 cho USV-VN từ đường vẽ tay (tools/annotate_water_edge.py) + nhãn YOLO.

    python tools/make_usv_masks.py                       # mọi split, ảnh nào đã có đủ 2 đường
    python tools/make_usv_masks.py --splits test --vis   # kèm ảnh chồng màu để kiểm tra bằng mắt

Ghi vào <usv-root>/<split>/:
    masks/<stem>m.png    0 vật cản · 1 nước · 2 trời · 4 không xác định (như MaSTr1325), cùng kích thước ảnh
    mask_list.txt        danh sách stem có mask — dùng làm image_list cho wars.data.mastr.MaSTr1325
    mask_vis/<stem>.jpg  (với --vis) ảnh chồng màu: vàng vật cản, xanh nước, tím trời, đen không xác định
Cách ghép: wars/data/usv_masks.py. Kết quả tái tạo được (GrabCut cố định seed), nên chỉ commit hai file
đường vẽ (water_edges.json, skylines.json), không commit mask.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.data.usv_masks import compose_mask, overlay_mask  # noqa: E402
from wars.data.usv_vn import label_path_for, list_images, load_water_edges, read_boxes  # noqa: E402

SPLITS = ("train", "valid", "test")
NAMES = {0: "vat can", 1: "nuoc", 2: "troi", 4: "khong xac dinh"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--water-edges", default="data/usv_vn/water_edges.json")
    ap.add_argument("--skylines", default="data/usv_vn/skylines.json")
    ap.add_argument("--band", type=int, default=3, help="nửa bề rộng dải 'không xác định' quanh ranh giới (px)")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    edges, skies = load_water_edges(args.water_edges), load_water_edges(args.skylines)
    root = Path(args.usv_root)
    for split in args.splits:
        images = list_images(root, split)
        if not images:
            continue
        out = root / split
        (out / "masks").mkdir(exist_ok=True)
        if args.vis:
            (out / "mask_vis").mkdir(exist_ok=True)
        done, missing, fallback, counts = [], [], 0, np.zeros(5)
        for p in images:
            if p.stem not in edges or p.stem not in skies:
                missing.append(p.stem)
                continue
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            mask, info = compose_mask(img, edges[p.stem], skies[p.stem], read_boxes(label_path_for(p), w, h),
                                      band=args.band)
            cv2.imwrite(str(out / "masks" / f"{p.stem}m.png"), mask)
            if args.vis:
                cv2.imwrite(str(out / "mask_vis" / f"{p.stem}.jpg"), overlay_mask(img, mask))
            done.append(p.stem)
            fallback += info["fallback"]
            counts += np.bincount(mask.ravel(), minlength=5)[:5]
        (out / "mask_list.txt").write_text("".join(f"{s}\n" for s in done))
        frac = counts / max(counts.sum(), 1)
        print(f"{split}: {len(done)}/{len(images)} ảnh có mask · thiếu đường: {len(missing)} · "
              f"vật thể dùng elip dự phòng: {fallback} · tỉ lệ lớp: "
              + ", ".join(f"{NAMES[c]} {frac[c]:.1%}" for c in NAMES))
        if missing:
            print(f"  chưa vẽ đủ 2 đường: {', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
