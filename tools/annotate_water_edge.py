"""Vẽ đường mép nước và đường chân trời cho USV-VN.

    python tools/annotate_water_edge.py --split test              # vẽ cả 2 đường (mặc định)
    python tools/annotate_water_edge.py --split test --only-edge  # chỉ mép nước

Hai đường, đều vẽ từ trái sang phải, điểm đầu/cuối nên chạm sát mép ảnh:
- Mép nước (vàng): ranh giới giữa mặt nước và bờ/vật cản tĩnh, vẽ *phía sau* vật cản động (coi như
  phao không có ở đó), giống MODS. Dùng cho FP/Precision/F1 (tools/evaluate_usv.py) và cho mask.
- Chân trời (xanh lơ): ranh giới giữa bờ (cây, nhà) và trời. Chỉ dùng để ghép mask
  (tools/make_usv_masks.py). Chỗ không có bờ thì vẽ trùng mép nước.
Bóng phản chiếu của bờ trên mặt nước tính là nước, nên mép nước nằm ở chân bờ, không ở chân bóng.

Phím: chuột trái = thêm điểm vào đường đang chọn · chuột phải = xóa điểm cuối · Tab/t = đổi đường ·
c = chép các đường của ảnh vừa lưu trước đó · r = xóa đường đang chọn · m = bật/tắt xem trước mask ·
h = ẩn/hiện khung nhãn YOLO ·
Enter/Space/n = lưu & ảnh sau · b = ảnh trước · q/Esc = lưu & thoát.
Hai file JSON được ghi sau mỗi ảnh nên có thể dừng và làm tiếp bất cứ lúc nào.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# OpenCV bản pip đi kèm Qt nhưng không kèm font → hàng loạt cảnh báo QFont. Trỏ Qt tới font hệ thống.
for _d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype", "/usr/share/fonts"):
    if os.path.isdir(_d):
        os.environ.setdefault("QT_QPA_FONTDIR", _d)
        break
# Thanh trạng thái Qt của OpenCV tính ra cỡ chữ âm → in "QFont::setPointSizeF ..." liên tục; tắt cảnh báo Qt
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;default.warning=false;qt.qpa.*=false")

import cv2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.data.usv_masks import compose_mask, overlay_mask  # noqa: E402
from wars.data.usv_vn import label_path_for, list_images, load_water_edges, read_boxes  # noqa: E402

LINES = ("edge", "sky")
COLORS = {"edge": (0, 255, 255), "sky": (255, 255, 0)}
LABELS = {"edge": "mep nuoc", "sky": "chan troi"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/usv_vn/water_edges.json", help="file mép nước")
    ap.add_argument("--skylines", default="data/usv_vn/skylines.json", help="file đường chân trời")
    ap.add_argument("--only-edge", action="store_true", help="chỉ vẽ mép nước")
    ap.add_argument("--redo", action="store_true", help="mở cả những ảnh đã vẽ")
    ap.add_argument("--no-boxes", action="store_true", help="không vẽ khung nhãn YOLO (bật lại bằng phím h)")
    args = ap.parse_args()

    files = {"edge": Path(args.out), "sky": Path(args.skylines)}
    store = {k: load_water_edges(f) for k, f in files.items()}
    active = ("edge",) if args.only_edge else LINES
    images = list_images(args.usv_root, args.split)

    def done(stem):
        return all(stem in store[k] for k in active)

    i = 0 if args.redo else next((k for k, p in enumerate(images) if not done(p.stem)), len(images))
    pts = {k: [] for k in LINES}
    cur = [active[0]]
    preview = [False]
    show_boxes = [not args.no_boxes]
    prev = {}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts[cur[0]].append([x, y])
        elif ev == cv2.EVENT_RBUTTONDOWN and pts[cur[0]]:
            pts[cur[0]].pop()

    cv2.namedWindow("water edge")
    cv2.setMouseCallback("water edge", on_mouse)
    while 0 <= i < len(images):
        p = images[i]
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        for k in LINES:
            pts[k][:] = [[int(round(x * w)), int(round(y * h))] for x, y in store[k].get(p.stem, [])]
        boxes = read_boxes(label_path_for(p), w, h)

        def norm(line):
            return [[x / w, y / h] for x, y in sorted(line)]

        cache = None
        while True:
            vis = img.copy()
            if preview[0] and not args.only_edge and all(len(pts[k]) >= 2 for k in LINES):
                key = tuple(tuple(map(tuple, pts[k])) for k in LINES)
                if cache is None or cache[0] != key:
                    mask, _ = compose_mask(img, norm(pts["edge"]), norm(pts["sky"]), boxes)
                    cache = (key, overlay_mask(img, mask))
                vis = cache[1].copy()
            for b in boxes if show_boxes[0] else []:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            for k in active:
                line = sorted(pts[k])
                thick = 2 if k == cur[0] else 1
                for a, b in zip(line, line[1:]):
                    cv2.line(vis, tuple(a), tuple(b), COLORS[k], thick)
                for q in line:
                    cv2.circle(vis, tuple(q), 4 if k == cur[0] else 2, COLORS[k], -1)
            n_done = sum(done(s.stem) for s in images)
            cv2.putText(vis, f"{p.stem}  [{i + 1}/{len(images)}]  da ve: {n_done}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"dang ve: {LABELS[cur[0]]}" + ("  (Tab doi, m xem mask)" if not args.only_edge else ""),
                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[cur[0]], 2)
            cv2.imshow("water edge", vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (9, ord("t")) and not args.only_edge:
                cur[0] = "sky" if cur[0] == "edge" else "edge"
            elif key == ord("c") and prev:
                for k in active:
                    if k in prev:
                        pts[k][:] = [list(q) for q in prev[k]]
            elif key == ord("r"):
                pts[cur[0]].clear()
            elif key == ord("m"):
                preview[0] = not preview[0]
            elif key == ord("h"):
                show_boxes[0] = not show_boxes[0]
            elif key in (13, 32, ord("n"), ord("b"), ord("q"), 27):
                saved = {k for k in active if len(pts[k]) >= 2}
                for k in saved:
                    store[k][p.stem] = norm(pts[k])
                    files[k].write_text(json.dumps(store[k], indent=1))
                if saved:
                    prev = {k: [list(q) for q in sorted(pts[k])] for k in saved}
                if key in (ord("q"), 27):
                    i = -1
                else:
                    i = max(i - 1, 0) if key == ord("b") else i + 1
                break
    cv2.destroyAllWindows()
    for k in active:
        print(f"Đã lưu {len(store[k])} đường {LABELS[k]} vào {files[k]}")


if __name__ == "__main__":
    main()
