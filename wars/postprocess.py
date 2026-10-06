"""Hậu xử lý dùng chung cho PyTorch, ONNX, Hailo emulator và Raspberry Pi (chỉ numpy + OpenCV)."""
from pathlib import Path

import cv2
import numpy as np

OBSTACLE, WATER, SKY = 0, 1, 2
# Màu mask theo cấu hình mặc định của MODS evaluator (WaSR predict_mods.py)
MODS_COLORS = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], np.uint8)
VIS_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)


def logits_to_mask(logits, out_hw):
    """Logits (3,h,w) ở độ phân giải 1/4 → mask lớp uint8 (H,W). Upsample logits rồi mới argmax."""
    up = cv2.resize(np.ascontiguousarray(logits.transpose(1, 2, 0), np.float32),
                    (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return up.argmax(-1).astype(np.uint8)


def save_mask(mask, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def load_mask(path, hw=None):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if hw is not None and m.shape != tuple(hw):
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def overlay(img_rgb, mask, alpha=0.5):
    color = VIS_COLORS[cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)]
    return (img_rgb * (1 - alpha) + color * alpha).astype(np.uint8)


def water_edge_from_mask(mask):
    """Ước lượng mép nước từ mask dự đoán: mỗi cột lấy hàng cao nhất của vùng nước lớn nhất
    chạm đáy ảnh. Trả về mảng (W,) chỉ số hàng; cột không có nước → H (không có mặt nước)."""
    h, w = mask.shape
    n, lab = cv2.connectedComponents((mask == WATER).astype(np.uint8), connectivity=4)
    bottom = lab[-1][lab[-1] > 0]
    edge = np.full(w, h, np.int32)
    if len(bottom) == 0:
        return edge
    water = lab == np.bincount(bottom).argmax()
    has = water.any(0)
    edge[has] = water.argmax(0)[has]
    return edge


def extract_obstacles(mask, min_area=20, hfov_deg=None, edge=None):
    """Tách các vật cản nằm trên mặt nước từ mask.

    Một thành phần liên thông lớp vật cản được giữ nếu đáy của nó nằm dưới mép nước
    (`edge` từ water_edge_from_mask nếu không truyền vào). Trả về list dict:
    bbox [x1,y1,x2,y2], area, bottom (x,y) và bearing_deg (nếu biết góc nhìn ngang camera).
    """
    h, w = mask.shape
    if edge is None:
        edge = water_edge_from_mask(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask == OBSTACLE).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        ys, xs = np.nonzero(lab[y:y + bh, x:x + bw] == i)
        by = y + ys.max()
        bx = x + int(np.median(xs[ys == ys.max()]))
        if by <= edge[bx]:
            continue  # nằm trên mép nước → bờ/vật cản tĩnh phía xa
        ob = {"bbox": [int(x), int(y), int(x + bw), int(y + bh)], "area": int(area), "bottom": [bx, int(by)]}
        if hfov_deg is not None:
            ob["bearing_deg"] = (bx / w - 0.5) * hfov_deg
        out.append(ob)
    return out
