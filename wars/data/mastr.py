from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .transforms import normalize


def read_mask(path):
    """Mask MaSTr1325 (0 vật cản, 1 nước, 2 trời, 4 không xác định) → one-hot float HxWx3.

    Pixel 'không xác định' thành one-hot toàn 0 và bị bỏ qua khi tính loss/metric.
    """
    m = np.array(Image.open(path))
    return np.stack([m == 0, m == 1, m == 2], axis=-1).astype(np.float32)


class MaSTr1325(torch.utils.data.Dataset):
    """MaSTr1325: `image_dir/XXXX.jpg` + `mask_dir/XXXXm.png`; danh sách ảnh trong `image_list`."""

    def __init__(self, image_dir, mask_dir, image_list, transform):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.names = [l.strip() for l in Path(image_list).read_text().splitlines() if l.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = np.array(Image.open(self.image_dir / f"{name}.jpg").convert("RGB"))
        mask = read_mask(self.mask_dir / f"{name}m.png")
        out = self.transform(image=img, mask=mask)
        seg = torch.from_numpy(out["mask"].transpose(2, 0, 1).copy())
        return normalize(out["image"]), seg


def read_mask_lars(path):
    """Mask semantic LaRS (0 vật cản, 1 nước, 2 trời, 255 bỏ qua) → one-hot float HxWx3; 255 thành toàn 0."""
    m = np.array(Image.open(path))
    return np.stack([m == 0, m == 1, m == 2], axis=-1).astype(np.float32)


class LaRS(torch.utils.data.Dataset):
    """LaRS (Žust và cs., ICCV 2023): `<root>/<split>/images/<name>.jpg` + `semantic_masks/<name>.png`,
    danh sách trong `<split>/image_list.txt`. LaRS train đã gồm 1323 ảnh MaSTr1325 (gán nhãn lại) + 1282 ảnh
    sông/hồ/biển khác, nên dùng thay cho MaSTr1325, không ghép thêm."""

    def __init__(self, root, split, transform):
        self.dir = Path(root) / split
        self.names = [l.strip() for l in (self.dir / "image_list.txt").read_text().splitlines() if l.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = np.array(Image.open(self.dir / "images" / f"{name}.jpg").convert("RGB"))
        mask = read_mask_lars(self.dir / "semantic_masks" / f"{name}.png")
        out = self.transform(image=img, mask=mask)
        seg = torch.from_numpy(out["mask"].transpose(2, 0, 1).copy())
        return normalize(out["image"]), seg
