"""Tự điền mép nước + đường chân trời cho khung chưa vẽ bằng nội suy giữa các khung mốc vẽ tay.

    python tools/annotate_water_edge.py --split test --step 10 --no-boxes   # vẽ tay khung mốc
    python tools/interpolate_edges.py --split test                          # điền phần còn lại
    python tools/annotate_water_edge.py --split test --review-auto --no-boxes   # duyệt / sửa

Không dùng mô hình nào → nhãn của tập test vẫn độc lập với các mô hình được đánh giá. Mỗi khung chưa vẽ
lấy hai khung mốc gần nhất trước/sau theo số thứ tự khung (left_XXXX), lấy mẫu lại hai polyline trên cùng
lưới x rồi nội suy tuyến tính theo thời gian; ngoài khoảng mốc thì chép khung mốc gần nhất. Khung tự điền
được ghi vào --auto-list (chưa duyệt) cho tới khi người duyệt bấm Enter trong annotate_water_edge.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.data.usv_vn import list_images, load_water_edges  # noqa: E402

GRID = np.linspace(0.0, 1.0, 33)


def resample(line):
    pts = np.array(sorted(line), np.float64)
    return np.interp(GRID, pts[:, 0], pts[:, 1])  # ngoài hai đầu: giữ y của điểm đầu/cuối


def frame_no(stem):
    return int(stem.rsplit("_", 1)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--split", default="test")
    ap.add_argument("--water-edges", default="data/usv_vn/water_edges.json")
    ap.add_argument("--skylines", default="data/usv_vn/skylines.json")
    ap.add_argument("--auto-list", default="data/usv_vn/edges_auto.json")
    ap.add_argument("--max-gap", type=int, default=40, help="khoảng cách khung tối đa tới khung mốc")
    args = ap.parse_args()

    files = {"edge": Path(args.water_edges), "sky": Path(args.skylines)}
    store = {k: load_water_edges(f) for k, f in files.items()}
    auto_path = Path(args.auto_list)
    auto = set(json.loads(auto_path.read_text())) if auto_path.exists() else set()
    stems = [p.stem for p in list_images(args.usv_root, args.split)]
    keys = sorted((frame_no(s), s) for s in stems if s not in auto and all(s in store[k] for k in store))
    if not keys:
        raise SystemExit("Chưa có khung mốc nào vẽ đủ 2 đường")
    kf = np.array([f for f, _ in keys])

    filled, skipped = [], []
    for s in stems:
        if s in [k for _, k in keys]:
            continue
        f = frame_no(s)
        j = int(np.searchsorted(kf, f))
        a = keys[j - 1] if j > 0 else None
        b = keys[j] if j < len(keys) else None
        near = min(abs(f - x[0]) for x in (a, b) if x)
        if near > args.max_gap:
            skipped.append(s)
            continue
        for k in store:
            if a and b:
                t = (f - a[0]) / (b[0] - a[0])
                y = (1 - t) * resample(store[k][a[1]]) + t * resample(store[k][b[1]])
            else:
                y = resample(store[k][(a or b)[1]])
            store[k][s] = [[float(x), float(v)] for x, v in zip(GRID, y)]
        auto.add(s)
        filled.append(s)

    for k, f in files.items():
        f.write_text(json.dumps(store[k], indent=1))
    auto_path.write_text(json.dumps(sorted(auto), indent=1))
    print(f"{args.split}: {len(keys)} khung mốc · tự điền {len(filled)} khung (chưa duyệt) · "
          f"bỏ qua {len(skipped)} khung cách mốc > {args.max_gap}")
    print(f"Duyệt: python tools/annotate_water_edge.py --split {args.split} --review-auto --no-boxes")


if __name__ == "__main__":
    main()
