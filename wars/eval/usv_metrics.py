"""Giao thức đánh giá vật cản trên USV-VN (chỉ có nhãn bounding box) — xem docs/de_cuong.md mục 8.

Hai chế độ:
- coverage (tiêu chí chính, kiểu MODS, dùng chung cho phân đoạn và YOLO — box YOLO được tô thành
  mask vật cản): GT là TP nếu tỉ lệ pixel vật cản dự đoán trong box ≥ `coverage_thr`.
  FP = thành phần liên thông vật cản trong vùng nước không chạm GT box nào.
- iou (tiêu chí phụ, chuẩn phát hiện đối tượng): dự đoán quy về box, ghép tham lam theo
  IoU ≥ `iou_thr`. FP = box chưa ghép có điểm giữa đáy nằm trong vùng nước. Bất lợi cho phân đoạn:
  camera USV-VN đặt sát mặt nước nên vật cản đè lên dải bờ trong ảnh, mask dính liền với bờ.

FP chỉ tính được khi có đường mép nước vẽ tay (vùng nước = phía dưới đường đó); nếu không, chỉ
báo cáo Recall. Ngưỡng `coverage_thr` mặc định 0.5 — cần đối chiếu với MODS evaluation toolkit
trước khi đưa số vào bài báo.
"""
import cv2
import numpy as np

from ..postprocess import OBSTACLE

# Ngưỡng kích thước theo căn bậc hai diện tích box (pixel). Mặc định theo COCO; trên USV-VN mọi box
# valid/test đều rơi vào "medium" nên nên truyền ngưỡng riêng (vd. --size-edges 48 72).
COCO_SIZE_EDGES = (32, 96)


def size_bins(edges=COCO_SIZE_EDGES):
    names = ("small", "medium", "large") if len(edges) == 2 else [f"bin{i}" for i in range(len(edges) + 1)]
    bounds = [0, *edges, float("inf")]
    return [(n, bounds[i] ** 2, bounds[i + 1] ** 2) for i, n in enumerate(names)]


def size_bin(box, bins=None):
    a = (box[2] - box[0]) * (box[3] - box[1])
    return next(name for name, lo, hi in (bins or size_bins()) if lo <= a < hi)


def water_region(edge_pts, h, w):
    """Polyline mép nước (tọa độ chuẩn hóa, trái → phải) → mask bool vùng phía dưới."""
    pts = np.array(edge_pts, np.float32) * [w, h]
    pts = pts[np.argsort(pts[:, 0])]
    poly = np.vstack([[0, pts[0, 1]], pts, [w - 1, pts[-1, 1]], [w - 1, h - 1], [0, h - 1]])
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [np.round(poly).astype(np.int32)], 1)
    return region.astype(bool)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _components(obst, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(obst.astype(np.uint8), connectivity=8)
    return [(lab == i, [stats[i, 0], stats[i, 1], stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]])
            for i in range(1, n) if stats[i, 4] >= min_area]


def eval_mask_coverage(mask, gt_boxes, water=None, coverage_thr=0.5, min_area=20):
    obst = mask == OBSTACLE
    hits = []
    for b in gt_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        roi = obst[max(y1, 0):y2, max(x1, 0):x2]
        hits.append(roi.size > 0 and roi.mean() >= coverage_thr)
    fp = None
    if water is not None:
        gt_mask = np.zeros_like(obst)
        for b in gt_boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            gt_mask[max(y1, 0):y2, max(x1, 0):x2] = True
        fp = sum(1 for comp, _ in _components(obst & water, min_area) if not (comp & gt_mask).any())
    return hits, fp


def boxes_to_mask(boxes, h, w):
    """Tô các box [x1,y1,x2,y2,...] thành mask lớp (vật cản bên trong box, phần còn lại = nước)."""
    mask = np.ones((h, w), np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        mask[max(y1, 0):y2, max(x1, 0):x2] = OBSTACLE
    return mask


def mask_to_boxes(mask, water=None, min_area=20):
    obst = mask == OBSTACLE
    if water is not None:
        obst &= water
    return [box + [1.0] for _, box in _components(obst, min_area)]


def eval_boxes_iou(pred_boxes, gt_boxes, water=None, iou_thr=0.5):
    """pred_boxes: list [x1,y1,x2,y2,score]."""
    preds = sorted(pred_boxes, key=lambda p: -p[4])
    hits, used = [False] * len(gt_boxes), set()
    for pi, p in enumerate(preds):
        best, bj = iou_thr, -1
        for j, g in enumerate(gt_boxes):
            if not hits[j] and (v := iou(p, g)) >= best:
                best, bj = v, j
        if bj >= 0:
            hits[bj] = True
            used.add(pi)
    fp = None
    if water is not None:
        h, w = water.shape
        fp = 0
        for pi, p in enumerate(preds):
            if pi in used:
                continue
            bx = int(np.clip((p[0] + p[2]) / 2, 0, w - 1))
            by = int(np.clip(p[3], 0, h - 1))
            fp += bool(water[by, bx])
    return hits, fp


class Accumulator:
    def __init__(self, size_edges=COCO_SIZE_EDGES):
        self.bins = size_bins(size_edges)
        self.tp = {k: 0 for k, *_ in self.bins}
        self.n = {k: 0 for k, *_ in self.bins}
        self.fp, self.fp_valid, self.images = 0, True, 0

    def add(self, gt_boxes, hits, fp):
        self.images += 1
        for b, h in zip(gt_boxes, hits):
            s = size_bin(b, self.bins)
            self.n[s] += 1
            self.tp[s] += int(h)
        if fp is None:
            self.fp_valid = False
        else:
            self.fp += fp

    def summary(self):
        tp, n = sum(self.tp.values()), sum(self.n.values())
        rec = tp / n if n else 0.0
        res = {"images": self.images, "gt": n, "tp": tp, "recall": rec,
               "recall_by_size": {k: (self.tp[k] / self.n[k] if self.n[k] else None) for k in self.n},
               "gt_by_size": dict(self.n)}
        if self.fp_valid:
            prec = tp / (tp + self.fp) if tp + self.fp else 0.0
            res.update(fp=self.fp, precision=prec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return res
