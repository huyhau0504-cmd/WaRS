"""Ghép mask 3 lớp kiểu MaSTr1325 cho USV-VN từ 2 đường vẽ tay + hộp YOLO có sẵn.

Không phụ thuộc torch (dùng chung cho tools/annotate_water_edge.py và tools/make_usv_masks.py).

- Phía trên đường chân trời (ranh giới bờ–trời) → trời (2).
- Phía dưới mép nước (ranh giới nước–bờ, vẽ phía sau vật cản động như MODS) → nước (1);
  bóng phản chiếu và sóng tính là nước, theo quy ước MaSTr1325.
- Giữa hai đường → vật cản tĩnh: bờ, cây, nhà (0).
- Vật thể trong hộp YOLO → tách bằng GrabCut khởi tạo từ hộp → vật cản (0).
- Dải ±`band` pixel quanh mọi ranh giới lớp → không xác định (4), bị bỏ qua khi huấn luyện/đánh giá.
"""
import cv2
import numpy as np

from ..eval.usv_metrics import water_region

OBSTACLE, WATER, SKY, UNKNOWN = 0, 1, 2, 4


def segment_object(img_bgr, box, margin=4, iters=5):
    """Tách vật thể trong hộp [x1,y1,x2,y2] bằng GrabCut. Trả về (mask bool, dùng_dự_phòng).

    Nếu GrabCut cho kết quả bất thường (gần rỗng hoặc gần đầy hộp) thì dùng hình elip nội tiếp hộp.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1 - margin, 0), max(y1 - margin, 0)
    x2, y2 = min(x2 + margin, w - 1), min(y2 + margin, h - 1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((h, w), bool), True
    cv2.setRNGSeed(0)  # GrabCut khởi tạo GMM bằng k-means → cố định để mask tái tạo được
    gc = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    except cv2.error:
        fg = np.zeros((h, w), bool)
    frac = fg[y1:y2, x1:x2].mean()
    if 0.1 <= frac <= 0.95:
        # Giữ thành phần liên thông lớn nhất để bỏ đốm nhiễu
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        if n > 1:
            fg = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return fg, False
    ell = np.zeros((h, w), np.uint8)
    cv2.ellipse(ell, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 1, -1)
    return ell.astype(bool), True


def compose_mask(img_bgr, edge_pts, sky_pts, boxes, band=3):
    """Mask uint8 (H,W) giá trị 0/1/2/4. `edge_pts`, `sky_pts`: polyline chuẩn hóa 0–1, trái → phải.

    Trả về (mask, info) với info = {"fallback": số vật thể phải dùng elip dự phòng}.
    """
    h, w = img_bgr.shape[:2]
    below_sky = water_region(sky_pts, h, w)
    water = water_region(edge_pts, h, w)
    mask = np.full((h, w), OBSTACLE, np.uint8)
    mask[~below_sky] = SKY
    mask[water] = WATER  # mép nước được ưu tiên nếu hai đường cắt nhau
    fallback = 0
    for b in boxes:
        obj, fb = segment_object(img_bgr, b)
        mask[obj] = OBSTACLE
        fallback += fb
    if band > 0:
        k = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
        boundary = cv2.dilate(mask, k) != cv2.erode(mask, k)
        mask[boundary] = UNKNOWN
    return mask, {"fallback": fallback}


VIS_COLORS = {OBSTACLE: (37, 195, 247), WATER: (224, 167, 41), SKY: (164, 75, 90), UNKNOWN: (0, 0, 0)}


def overlay_mask(img_bgr, mask, alpha=0.45):
    """Ảnh BGR chồng màu lớp (vàng vật cản, xanh nước, tím trời, đen không xác định)."""
    color = np.zeros_like(img_bgr)
    for c, bgr in VIS_COLORS.items():
        color[mask == c] = bgr
    return (img_bgr * (1 - alpha) + color * alpha).astype(np.uint8)
