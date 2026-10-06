"""Chia lại bộ dữ liệu USV (YOLO format) theo đoạn thời gian để tránh rò rỉ giữa train/valid/test.

Ảnh là các khung hình liên tiếp `left_XXXX.jpg`; chia ngẫu nhiên khiến các khung gần giống hệt nhau
rơi vào cả train lẫn test. Ranh giới được đặt tại các khoảng trống tự nhiên trong chuỗi khung hình.

Cách chia gốc (Roboflow, ngẫu nhiên) được lưu vào splits/original_split.csv trước lần chia đầu tiên.
Khôi phục cách chia gốc:  python tools/resplit_by_time.py --restore
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "usv_vn"
SPLITS = ("train", "valid", "test")
ORIGINAL_CSV = ROOT / "splits" / "original_split.csv"
TEMPORAL_CSV = ROOT / "splits" / "temporal_split.csv"

# (split, khung đầu, khung cuối) — khoảng trống 755–862 tách train/valid, 1048–1058 tách valid/test
TEMPORAL_RANGES = [("train", 1, 754), ("valid", 755, 1047), ("test", 1048, 1259)]


def frame_id(name: str) -> int:
    return int(re.search(r"(\d+)", name).group(1))


def current_assignment() -> dict[str, str]:
    return {img.stem: s for s in SPLITS for img in (DATA / s / "images").glob("*.jpg")}


def move_to(assign: dict[str, str]) -> None:
    for s in SPLITS:
        for sub in ("images", "labels"):
            (DATA / s / sub).mkdir(parents=True, exist_ok=True)
    cur = current_assignment()
    for stem, old in cur.items():
        new = assign[stem]
        if new == old:
            continue
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            shutil.move(DATA / old / sub / f"{stem}{ext}", DATA / new / sub / f"{stem}{ext}")


def write_csv(path: Path, assign: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "frame", "split"])
        for stem in sorted(assign, key=frame_id):
            w.writerow([stem, frame_id(stem), assign[stem]])


def read_csv(path: Path) -> dict[str, str]:
    with path.open() as f:
        return {r["stem"]: r["split"] for r in csv.DictReader(f)}


def temporal_assignment(stems) -> dict[str, str]:
    out = {}
    for stem in stems:
        k = frame_id(stem)
        out[stem] = next(s for s, a, b in TEMPORAL_RANGES if a <= k <= b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true", help="khôi phục cách chia gốc")
    args = ap.parse_args()

    cur = current_assignment()
    if args.restore:
        move_to(read_csv(ORIGINAL_CSV))
    else:
        if not ORIGINAL_CSV.exists():
            write_csv(ORIGINAL_CSV, cur)
        assign = temporal_assignment(cur)
        move_to(assign)
        write_csv(TEMPORAL_CSV, assign)

    after = current_assignment()
    for s in SPLITS:
        ks = sorted(frame_id(k) for k, v in after.items() if v == s)
        print(f"{s:5s}: {len(ks):4d} ảnh, khung {ks[0]}–{ks[-1]}")


if __name__ == "__main__":
    main()
