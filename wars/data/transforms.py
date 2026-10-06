import albumentations as A
import cv2
import numpy as np
import torch

from ..constants import MEAN, STD


def normalize(img_rgb_uint8):
    """HWC uint8 RGB → CHW float tensor đã chuẩn hóa."""
    x = (img_rgb_uint8.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def train_transform(size, vn_color=False):
    """Augmentation của eWaSR (+ tùy chọn `vn_color` mô phỏng ảnh ám màu của dữ liệu USV-VN).

    Viền ngoài ảnh sau Affine được tô 0 ở mask one-hot → pixel bị bỏ qua khi tính loss.
    """
    h, w = size
    color = [A.ColorJitter(p=0.7, hue=0.05), A.RandomGamma(p=1, gamma_limit=(70, 120))]
    if vn_color:
        color += [A.RGBShift(r_shift_limit=40, g_shift_limit=20, b_shift_limit=40, p=0.7),
                  A.RandomBrightnessContrast(p=0.5)]
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(),
        A.Affine(scale=(1.0, 1.3), rotate=(-15, 15), translate_percent=(-0.0625, 0.0625), p=0.7),
        A.Compose(color, p=0.5),
        A.Compose([A.GaussNoise(p=0.5), A.ISONoise(p=0.5)], p=0.3),
    ])


def eval_transform(size):
    h, w = size
    return A.Compose([A.Resize(h, w, interpolation=cv2.INTER_AREA)])
