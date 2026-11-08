"""Hình định tính cho bài báo (Fig. 2): MODS INT8 trên Pi (E7 so với E1 ở bộ calib bị sập) và một cảnh sông LaRS
(E7 huấn luyện trên MaSTr1325 so với trên LaRS). Chỉ đọc mask đã có trong preds/, không chạy mô hình.

    python tools/make_qual_figure.py --out paper/figs/qual.png
    python tools/make_qual_figure.py --list-mods 8      # in các khung E1 có nhiều báo động sai vùng nguy hiểm nhất

Lớp: vật cản vàng, nước xanh, trời tím (VIS_COLORS), phủ 45% lên ảnh.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.postprocess import MODS_COLORS, VIS_COLORS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def mods_seq_dirs(mods_root):
    # "id" trong <out>_frames.json của toolkit là vị trí (từ 1) trong danh sách mods.json, không phải trường "id"
    seqs = json.loads((mods_root / "mods.json").read_text())["dataset"]["sequences"]
    return {i + 1: s["path"].strip("/").split("/")[0] for i, s in enumerate(seqs)}


def read_mods_mask(path):
    """Mask màu MODS (BGR trên đĩa) → mã lớp 0/1/2."""
    rgb = cv2.imread(str(path))[:, :, ::-1]
    cls = np.full(rgb.shape[:2], 255, np.uint8)
    for c, col in enumerate(MODS_COLORS):
        cls[(rgb == col).all(-1)] = c
    return cls


def overlay(img, cls, alpha=0.45):
    cls = cv2.resize(cls, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    col = np.zeros_like(img)
    valid = cls < 3
    col[valid] = VIS_COLORS[cls[valid]]
    out = img.copy()
    out[valid] = (img[valid] * (1 - alpha) + col[valid] * alpha).astype(np.uint8)
    return out


def label(img, text):
    out = img.copy()
    h = 28
    cv2.rectangle(out, (0, 0), (out.shape[1], h + 6), (0, 0, 0), -1)
    cv2.putText(out, text, (6, h), cv2.FONT_HERSHEY_SIMPLEX, h / 30, (255, 255, 255), max(1, h // 15), cv2.LINE_AA)
    return out


def worst_mods_frames(bad, good, n):
    a = json.loads((ROOT / f"results/mods/{bad}_frames.json").read_text())
    b = json.loads((ROOT / f"results/mods/{good}_frames.json").read_text())
    rows = []
    for sa, sb in zip(a["sequences"], b["sequences"]):
        for fa, fb in zip(sa["frames"], sb["frames"]):
            rows.append((len(fa["obstacles_danger"]["fp_list"]) - len(fb["obstacles_danger"]["fp_list"]),
                         len(fa["obstacles_danger"]["fp_list"]), len(fb["obstacles_danger"]["fp_list"]),
                         sa["id"], fa["img_name"]))
    return sorted(rows, reverse=True)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods-root", default=str(Path.home() / "datasets/mods"))
    ap.add_argument("--lars-root", default=str(Path.home() / "datasets/lars"))
    ap.add_argument("--good", default="e7_ewasr_h_lite_c64noeqs2max_int8")
    ap.add_argument("--bad", default="e1_ewasr_c64noeqs2max_int8")
    ap.add_argument("--mods-frame", default="30/00025950L", help="<id sequence>/<khung>")
    ap.add_argument("--lars-frame", default="orca_usv_inland_N03_3_00777")
    ap.add_argument("--lars-a", default="e7_ewasr_h_lite_fp32")
    ap.add_argument("--lars-b", default="e7_ewasr_h_lite_lars_fp32")
    ap.add_argument("--width", type=int, default=700, help="chiều rộng mỗi ô (px)")
    ap.add_argument("--list-mods", type=int)
    ap.add_argument("--out", default=str(ROOT / "paper/figs/qual.png"))
    args = ap.parse_args()

    if args.list_mods:
        for r in worst_mods_frames(args.bad, args.good, args.list_mods):
            print("chênh %d (bad %d, good %d) · %d/%s" % r)
        return

    mods = Path(args.mods_root).expanduser()
    sid, frame = args.mods_frame.split("/")
    seq = mods_seq_dirs(mods)[int(sid)]
    img = cv2.imread(str(mods / "sequences" / seq / "frames" / f"{frame}.jpg"))[:, :, ::-1]
    row1 = [(img, "MODS input"),
            (overlay(img, read_mods_mask(ROOT / "preds/mods" / args.good / seq / f"{frame}.png")), "eWaSR-Lite INT8"),
            (overlay(img, read_mods_mask(ROOT / "preds/mods" / args.bad / seq / f"{frame}.png")), "eWaSR INT8")]

    lars = Path(args.lars_root).expanduser() / "val"
    img = cv2.imread(str(lars / "images" / f"{args.lars_frame}.jpg"))[:, :, ::-1]
    m = lambda r: cv2.imread(str(ROOT / "preds/lars" / r / "masks" / f"{args.lars_frame}.png"), cv2.IMREAD_GRAYSCALE)
    row2 = [(img, "LaRS river input"),
            (overlay(img, m(args.lars_a)), "trained on MaSTr1325"),
            (overlay(img, m(args.lars_b)), "trained on LaRS")]

    w = args.width
    fit = lambda t: cv2.resize(t, (w, int(round(t.shape[0] * w / t.shape[1]))), interpolation=cv2.INTER_AREA)
    rows = [np.hstack([label(fit(t), text) for t, text in r]) for r in (row1, row2)]
    gap = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    fig = np.vstack([rows[0], gap, rows[1]])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fig[:, :, ::-1])
    print(f"{seq}/{frame} + {args.lars_frame} → {out} ({fig.shape[1]}×{fig.shape[0]})")


if __name__ == "__main__":
    main()
