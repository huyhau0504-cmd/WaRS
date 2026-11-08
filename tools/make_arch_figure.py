"""Sơ đồ kiến trúc eWaSR-Lite cho bài báo (Fig. 1) → paper/figs/arch.pdf (+ .png để xem nhanh).

    third_party/.venv-mods/bin/python tools/make_arch_figure.py

Cần matplotlib (venv chính không có; venv của MODS toolkit có). Nội dung theo docs/design.md §3 và
wars/models/ewasr.py: ResNet-18 → PPA (6×8 token, 960 kênh) → feature mixer CCCCDD, chỉ khối 0 có MLP →
SIM×4 (nhánh 1/8 qua enricher DD) → head ở H/4; upsample + argmax trên CPU.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, KEEP, NEW, GONE, CHIP, HOST = "#222222", "#9a9a9a", "#e8eef7", "#f6c64a", "#ffffff", "#f3f3f3", "#fbf3e4"


def box(ax, x, y, w, h, text, fc=KEEP, ec=INK, ls="-", fs=6.5, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/figs/arch.pdf"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.16, 2.15))
    ax.set_xlim(0, 21.5)
    ax.set_ylim(0, 6.45)
    ax.axis("off")

    # vùng chạy trên Hailo-8L và trên CPU
    ax.add_patch(FancyBboxPatch((1.55, 0.15), 15.75, 6.1, boxstyle="round,pad=0,rounding_size=0.15", fc=CHIP, ec=GREY, lw=0.6))
    ax.text(1.75, 5.98, "Hailo-8L (INT8)", fontsize=6.5, color="#555555", va="center", style="italic")
    ax.add_patch(FancyBboxPatch((17.5, 0.15), 3.9, 6.1, boxstyle="round,pad=0,rounding_size=0.15", fc=HOST, ec=GREY, lw=0.6))
    ax.text(17.7, 5.98, "Raspberry Pi 5 CPU", fontsize=6.5, color="#555555", va="center", style="italic")

    # đầu vào
    box(ax, 0.05, 2.55, 1.3, 1.3, "RGB\nuint8\n384×\n512", fc="#ffffff", fs=5.5)
    arrow(ax, 1.35, 3.2, 1.85, 3.2)

    # encoder ResNet-18: 4 mức
    ys = [4.7, 3.55, 2.4, 1.25]
    names = ["layer1  1/4", "layer2  1/8", "layer3  1/16", "layer4  1/32"]
    box(ax, 1.85, 0.75, 0.55, 4.9, "", fc=NEW)
    ax.text(2.125, 3.2, "ImageNet normalization", rotation=90, ha="center", va="center", fontsize=5.5)
    for y, n in zip(ys, names):
        box(ax, 2.6, y - 0.4, 1.9, 0.8, n, fs=6)
    ax.text(3.55, 0.45, "ResNet-18", ha="center", fontsize=6.5, fontweight="bold")
    arrow(ax, 2.4, 3.2, 2.6, 3.2)

    # PPA
    box(ax, 5.0, 1.0, 1.35, 4.5, "pyramid\npool,\nfixed\nkernels\n\n6×8\ntokens,\n960 ch", fs=5.6)
    for y in ys:
        arrow(ax, 4.5, y, 5.0, y if 1.0 < y < 5.5 else 3.2)

    # feature mixer: 6 khối MetaFormer
    ax.text(10.15, 5.6, "feature mixer (6 MetaFormer blocks)", ha="center", fontsize=6.5, fontweight="bold")
    mix = ["C", "C", "C", "C", "D", "D"]
    x0, bw, gap = 6.85, 1.07, 0.16
    for i, m in enumerate(mix):
        x = x0 + i * (bw + gap)
        box(ax, x, 3.35, bw, 1.6, f"{m}\nmixer", fc=NEW if m == "D" else KEEP, fs=6.5)
        if i == 0:
            box(ax, x, 1.55, bw, 1.4, "MLP\n960→\n3840→\n960", fs=5)
        else:
            box(ax, x, 1.55, bw, 1.4, "no\nMLP", fc=GONE, ec=GREY, ls=(0, (2, 1.5)), fs=5, color=GREY)
        ax.text(x + bw / 2, 1.2, str(i), ha="center", fontsize=5.5, color="#555555")
        if i:
            arrow(ax, x - gap, 4.15, x, 4.15)
    arrow(ax, 6.35, 3.2, 6.85, 4.15)
    xe = x0 + 6 * (bw + gap) - gap  # cuối mixer

    # SIM ×4 + enricher
    box(ax, 14.45, 3.3, 1.5, 2.1, "semantic\ninjection\nmodules\n×4", fs=5.6)
    arrow(ax, xe, 4.15, 14.45, 4.35)
    box(ax, 14.45, 0.75, 1.5, 1.3, "enricher\nD D", fc=NEW, fs=5.6)
    # nhánh skip đi vòng dưới feature mixer: 1/8 → enricher, các mức khác → SIM
    ax.plot([4.5, 4.78, 4.78, 15.2], [3.55, 3.55, 0.42, 0.42], color=GREY, lw=0.6)
    arrow(ax, 15.2, 0.42, 15.2, 0.75, color=GREY)
    ax.text(10.4, 0.6, "skip features: 1/8 via enricher; 1/4, 1/16, 1/32 directly", ha="center", fontsize=5.3, color="#555555")
    arrow(ax, 15.2, 2.05, 15.2, 3.3)

    # head
    box(ax, 16.2, 2.3, 1.0, 1.9, "head\n\nlogits\n3×96\n×128", fs=5.5)
    arrow(ax, 15.95, 3.6, 16.2, 3.4)

    # CPU: upsample + argmax + obstacle list
    arrow(ax, 17.2, 3.25, 17.7, 3.25)
    box(ax, 17.7, 2.5, 1.65, 1.5, "bilinear\nupsample\n& argmax", fc=NEW, fs=5.3)
    arrow(ax, 19.35, 3.25, 19.7, 3.25)
    box(ax, 19.7, 2.4, 1.6, 1.7, "mask,\nobstacle\nlist,\nbearings", fc="#ffffff", fs=5.5)

    # chú thích
    lx = 17.75
    for k, (fc, ec, ls, t) in enumerate([(NEW, INK, "-", "changed vs. eWaSR"), (GONE, GREY, (0, (2, 1.5)), "MLP removed")]):
        y = 1.15 - k * 0.55
        ax.add_patch(FancyBboxPatch((lx, y - 0.17), 0.42, 0.34, boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=0.6, ls=ls))
        ax.text(lx + 0.58, y, t, fontsize=5.5, va="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.01)
    print(f"→ {out}, {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
