"""Phân bố kích hoạt FP32 theo lớp trên đúng tập ảnh calibration của hailo/compile.py — kiểm tra giả thuyết
"MLP còn sống của feature mixer có ngoại lai, nên calib nhiều ảnh hơn làm khoảng lượng tử rộng ra" (Q-011).

    python tools/activation_stats.py --run E1=ewasr:runs/e1_ewasr/final.pth E6=ewasr_lite:runs/e6_ewasr_lite/final.pth \\
        --calib-dir data/mastr1325/images --out docs/data/activation_stats.csv

Với mỗi lớp lá (Conv/BN/kích hoạt...) và mỗi khối MetaFormer, ghi:
    max64, max1024   max |x| trên tập calib 64 và 1024 ảnh (chọn y như load_calib(seed=0) của hailo/compile.py)
    p999             trung vị theo ảnh của phân vị 99,9% |x| (trên tập 1024)
    ratio            max1024 / p999 (độ "ngoại lai"); growth = max1024 / max64
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.constants import MEAN, STD  # noqa: E402
from wars.models import get_model, load_weights  # noqa: E402
from wars.models.layers import MetaFormerBlock  # noqa: E402


def calib_files(calib_dirs, n, seed=0):
    files = sorted(f for d in calib_dirs for f in Path(d).glob("*.jpg"))
    return list(np.random.default_rng(seed).choice(files, size=min(n, len(files)), replace=False))


def load(f, hw):
    img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = (img - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="TÊN=model:checkpoint")
    ap.add_argument("--calib-dir", nargs="+", default=["data/mastr1325/images"])
    ap.add_argument("--size", type=int, nargs=2, default=[384, 512])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--n-small", type=int, default=64)
    ap.add_argument("--out", default="docs/data/activation_stats.csv")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    big = calib_files(args.calib_dir, args.n)
    small = {str(f) for f in calib_files(args.calib_dir, args.n_small)}
    files = big + [f for f in calib_files(args.calib_dir, args.n_small) if str(f) not in {str(b) for b in big}]
    rows = []
    for spec in args.run:
        tag, rest = spec.split("=", 1)
        name, ckpt = rest.split(":", 1)
        model = get_model(name, pretrained_backbone=False)
        load_weights(model, ckpt)
        model.eval()
        stats, cur = {}, {}

        def hook(lname):
            def fn(mod, inp, out):
                if not torch.is_tensor(out) or out.dim() < 2:
                    return
                a = out.detach().abs().flatten()
                k = max(1, int(a.numel() * 0.001))
                top = torch.topk(a, k + 1).values
                mx, q = top[0].item(), top[-1].item()
                if lname in cur:  # module dùng lại nhiều lần trong một lượt forward
                    mx, q = max(mx, cur[lname][0]), max(q, cur[lname][1])
                cur[lname] = (mx, q)
            return fn

        names = []
        for lname, m in model.named_modules():
            if lname and (len(list(m.children())) == 0 or isinstance(m, MetaFormerBlock)):
                m.register_forward_hook(hook(lname))
                names.append(lname)
        in_big = {str(f) for f in big}
        with torch.no_grad():
            for i, f in enumerate(files):
                cur.clear()
                model(load(f, args.size))
                for lname, (mx, p) in cur.items():
                    s = stats.setdefault(lname, {"max1024": 0.0, "max64": 0.0, "p999": []})
                    if str(f) in in_big:
                        s["max1024"] = max(s["max1024"], mx)
                        s["p999"].append(p)
                    if str(f) in small:
                        s["max64"] = max(s["max64"], mx)
                if i % 128 == 0:
                    print(f"[{tag}] {i}/{len(files)}", flush=True)
        for lname in names:
            if lname not in stats:
                continue
            s = stats[lname]
            p = float(np.median(s["p999"]))
            rows.append([tag, name, lname, f"{s['max64']:.4g}", f"{s['max1024']:.4g}", f"{p:.4g}",
                         f"{s['max1024'] / max(p, 1e-12):.3g}", f"{s['max1024'] / max(s['max64'], 1e-12):.3g}"])
        print(f"[{tag}] xong {len(stats)} lớp", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "model", "layer", "max64", "max1024", "p999", "ratio", "growth"])
        w.writerows(rows)
    print(f"→ {out} ({len(rows)} dòng)")


if __name__ == "__main__":
    main()
