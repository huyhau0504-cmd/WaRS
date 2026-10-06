"""Vẽ đường mép nước cho USV-VN (cần cho FP/Precision/F1 trong tools/evaluate_usv.py).

    python tools/annotate_water_edge.py --split test

Quy ước (giống MODS): mép nước là ranh giới giữa mặt nước và bờ/vật cản tĩnh, vẽ từ trái sang
phải, đi *phía sau* vật cản động (coi như chai/rác không có ở đó).

Phím: chuột trái = thêm điểm · chuột phải = xóa điểm cuối · c = chép đường của ảnh trước ·
r = xóa hết · Enter/Space/n = lưu & ảnh sau · b = ảnh trước · q/Esc = lưu & thoát.
File JSON được ghi sau mỗi ảnh nên có thể dừng và làm tiếp bất cứ lúc nào.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.data.usv_vn import label_path_for, list_images, load_water_edges, read_boxes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/usv_vn/water_edges.json")
    ap.add_argument("--redo", action="store_true", help="mở cả những ảnh đã vẽ")
    args = ap.parse_args()

    edges = load_water_edges(args.out)
    images = list_images(args.usv_root, args.split)
    i = 0 if args.redo else next((k for k, p in enumerate(images) if p.stem not in edges), len(images))
    pts, prev = [], None

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow("water edge")
    cv2.setMouseCallback("water edge", on_mouse)
    while 0 <= i < len(images):
        p = images[i]
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        pts[:] = [[x * w, y * h] for x, y in edges.get(p.stem, [])]
        pts[:] = [[int(x), int(y)] for x, y in pts]
        boxes = read_boxes(label_path_for(p), w, h)
        while True:
            vis = img.copy()
            for b in boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            line = sorted(pts)
            for a, b in zip(line, line[1:]):
                cv2.line(vis, tuple(a), tuple(b), (0, 255, 255), 2)
            for q in line:
                cv2.circle(vis, tuple(q), 4, (0, 255, 0), -1)
            done = sum(s.stem in edges for s in images)
            cv2.putText(vis, f"{p.stem}  [{i + 1}/{len(images)}]  da ve: {done}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("water edge", vis)
            k = cv2.waitKey(20) & 0xFF
            if k == ord("c") and prev:
                pts[:] = [list(q) for q in prev]
            elif k == ord("r"):
                pts.clear()
            elif k in (13, 32, ord("n"), ord("b"), ord("q"), 27):
                if len(pts) >= 2:
                    edges[p.stem] = [[x / w, y / h] for x, y in sorted(pts)]
                    prev = [list(q) for q in sorted(pts)]
                    Path(args.out).write_text(json.dumps(edges, indent=1))
                if k in (ord("q"), 27):
                    i = -1
                else:
                    i = max(i - 1, 0) if k == ord("b") else i + 1
                break
    cv2.destroyAllWindows()
    print(f"Đã lưu {len(edges)} đường mép nước vào {args.out}")


if __name__ == "__main__":
    main()
