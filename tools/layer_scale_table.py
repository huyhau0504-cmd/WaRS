"""Bảng |γ| (layer_scale) theo khối MetaFormer của các bộ trọng số — bằng chứng MLP của feature mixer "chết".

    python tools/layer_scale_table.py --weights chinh_thuc=weights/ewasr_resnet18.pth \\
        E1=runs/e1_ewasr/final.pth E2=runs/e2_ewasr_h/final.pth --out docs/data/layer_scale.csv

Mỗi dòng: khối (mix.i = feature mixer, enr.i = long-skip enricher), nhánh (γ1 token mixer, γ2 MLP), rồi
trung bình |γ| theo kênh cho từng bộ trọng số. Khối không có nhánh (mixer-lite) để trống.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="tên=đường_dẫn.pth")
    ap.add_argument("--out", help="ghi CSV (mặc định chỉ in ra màn hình)")
    args = ap.parse_args()

    sds = {}
    for item in args.weights:
        name, path = item.split("=", 1)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sds[name] = sd.get("model", sd)
    keys = sorted({k for sd in sds.values() for k in sd if "layer_scale" in k},
                  key=lambda k: ("skip2" in k, int(k.split(".")[2]), k[-1]))
    rows = []
    for k in keys:
        part = k.split(".")
        block = ("enr" if "skip2" in part[1] else "mix") + f".{part[2]}"
        rows.append([block, "γ" + k[-1], *[f"{sd[k].abs().mean().item():.3e}" if k in sd else "" for sd in sds.values()]])
    header = ["khoi", "nhanh", *sds]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    w = csv.writer(sys.stdout if not args.out else open(Path(args.out), "w", newline=""))
    w.writerow(header)
    w.writerows(rows)
    if args.out:
        print(f"Đã ghi {args.out} ({len(rows)} dòng)")


if __name__ == "__main__":
    main()
