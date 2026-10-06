"""Bộ dữ liệu USV-VN tự thu: định dạng YOLO (`<split>/images/*.jpg`, `<split>/labels/*.txt`).

Không phụ thuộc torch để dùng được cả trên Raspberry Pi.
"""
import json
from pathlib import Path


def list_images(root, split):
    return sorted((Path(root) / split / "images").glob("*.jpg"))


def read_boxes(label_path, w, h):
    """Nhãn YOLO (cls cx cy bw bh, chuẩn hóa) → list [x1, y1, x2, y2] theo pixel."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, bw, bh = map(float, line.split()[:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def label_path_for(img_path):
    img_path = Path(img_path)
    return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"


def load_water_edges(path):
    """Đường mép nước vẽ tay: JSON {stem: [[x, y], ...]} với tọa độ chuẩn hóa 0–1, trái → phải.

    Trả về {} nếu chưa có file — khi đó bộ đánh giá chỉ tính được Recall, không tính FP.
    """
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}
