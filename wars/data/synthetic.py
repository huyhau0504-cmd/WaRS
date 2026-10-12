"""Dữ liệu giả theo bố cục MaSTr1325 cho smoke test (laptop không có MaSTr1325).

Không phụ thuộc torch.
"""
from pathlib import Path

import cv2
import numpy as np

# Màu RGB gần đúng cho từng lớp: 0 vật cản, 1 nước, 2 trời
_COLORS = {0: (150, 90, 40), 1: (40, 90, 140), 2: (190, 210, 235)}


def make_mastr_like(root, n=8, size=(384, 512), seed=0):
    """Tạo `images/XXXX.jpg`, `masks/XXXXm.png`, `train.txt`, `val.txt` (2 ảnh cuối là val).

    Mask: trời (2) phía trên, nước (1) phía dưới đường chân trời ngẫu nhiên, 1–3 hình chữ nhật
    vật cản (0) và một dải nhỏ giá trị 4 (không xác định). Trả về dict các đường dẫn dùng cho
    `--set data.*` của tools/train.py.
    """
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = size
    names = [f"{i:04d}" for i in range(1, n + 1)]
    for name in names:
        mask = np.full((h, w), 1, np.uint8)
        horizon = int(rng.integers(h // 4, h // 2))
        mask[:horizon] = 2
        for _ in range(int(rng.integers(1, 4))):
            bw, bh = int(rng.integers(w // 16, w // 5)), int(rng.integers(h // 16, h // 5))
            x, y = int(rng.integers(0, w - bw)), int(rng.integers(horizon - bh // 2, h - bh))
            mask[max(y, 0):y + bh, x:x + bw] = 0
        mask[-4:] = 4
        img = np.zeros((h, w, 3), np.float32)
        for c, rgb in _COLORS.items():
            img[mask == c] = rgb
        img[mask == 4] = _COLORS[1]
        img += rng.normal(0, 12, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(root / "images" / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "masks" / f"{name}m.png"), mask)
    n_val = 2 if n > 2 else 0
    (root / "train.txt").write_text("\n".join(names[:n - n_val]) + "\n")
    (root / "val.txt").write_text("\n".join(names[n - n_val:] or names) + "\n")
    return {"mastr_root": str(root), "image_dir": "images", "mask_dir": "masks",
            "train_list": str(root / "train.txt"), "val_list": str(root / "val.txt")}
