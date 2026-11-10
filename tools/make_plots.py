"""Đồ thị cho bài báo (bản EEJ) → paper/figs/{tradeoff,calib,gamma}.png (400 dpi) + .pdf.

    third_party/.venv-mods/bin/python tools/make_plots.py

Nguồn: results/mods/*_int8.json, *_fp32.json (MODS trên Pi / laptop), docs/data/hailo_calib_seeds.csv (FPS chip theo
từng bộ calib), docs/data/hailo_main_table.csv (E2, E6), docs/data/layer_scale_seeds.csv (γ₂). Dòng thiếu file kết quả
thì bỏ qua (in cảnh báo), nên chạy lại được khi có thêm số.

Màu: 3 ô đầu của bảng màu phân loại đã kiểm định (dataviz skill, palette.md: hợp lệ cho scatter với ≤3 nhóm) —
xanh = eWaSR (E1), cam = eWaSR-Lite (E7), xanh ngọc = trọng số công bố; E2/E6 màu xám kèm nhãn trực tiếp.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figs"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, MUTED, GRID = "#222222", "#555555", "#e6e6e3"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def mods(name):
    f = ROOT / "results/mods" / f"{name}.json"
    if not f.exists():
        print(f"  (thiếu {f.name})")
        return None
    d = json.loads(f.read_text())
    return {"dz": 100 * d["danger_zone"]["f1"], "dz_pr": 100 * d["danger_zone"]["precision"], "f1": 100 * d["obstacles"]["f1"]}


def hw_fps():
    rows = {r["hef"]: float(r["hw_fps"]) for r in csv.DictReader(open(ROOT / "docs/data/hailo_calib_seeds.csv"))}
    for r in csv.DictReader(open(ROOT / "docs/data/hailo_main_table.csv")):
        rows.setdefault(Path(r["hef"]).stem, float(r["hw_fps"]))
    return rows


# (nhãn, màu, marker, mặt đặc?, [(HEF, kết quả INT8)], kết quả FP32)
GROUPS = [
    ("eWaSR (E1)", BLUE, "o", True,
     [("e1_ewasr_c64noeqmax", "e1_ewasr_c64noeqmax_int8"), ("e1_ewasr_c64noeqs1max", "e1_ewasr_c64noeqs1max_int8"),
      ("e1_ewasr_c64noeqs2max", "e1_ewasr_c64noeqs2max_int8")], "e1_ewasr_fp32"),
    ("eWaSR (E1), equalization on", BLUE, "o", False,
     [("e1_ewasr_o0max", "e1_ewasr_o0max_int8"), ("e1_ewasr_c64eqs1max", "e1_ewasr_c64eqs1max_int8"),
      ("e1_ewasr_c64eqs2max", "e1_ewasr_c64eqs2max_int8")], "e1_ewasr_fp32"),
    ("eWaSR, D mixer (E2)", GREY, "s", True, [("e2_ewasr_h_c64noeqmax", "e2_ewasr_h_c64noeqmax_int8")], "e2_ewasr_h_fp32"),
    ("eWaSR-Lite, S mixer (E6)", GREY, "^", True,
     [("e6_ewasr_lite_c64noeqmax", "e6_ewasr_lite_c64noeqmax_int8")], "e6_ewasr_lite_fp32"),
    ("eWaSR-Lite (E7)", ORANGE, "D", True,
     [("e7_ewasr_h_lite_c64noeqmax", "e7_ewasr_h_lite_c64noeqmax_int8"),
      ("e7_ewasr_h_lite_c64noeqs1max", "e7_ewasr_h_lite_c64noeqs1max_int8"),
      ("e7_ewasr_h_lite_c64noeqs2max", "e7_ewasr_h_lite_c64noeqs2max_int8")], "e7_ewasr_h_lite_fp32"),
    ("eWaSR, published weights", AQUA, "P", True,
     [("ewasr_official_c64noeqmax", "ewasr_official_c64noeqmax_int8"),
      ("ewasr_official_c64noeqs1max", "ewasr_official_c64noeqs1max_int8"),
      ("ewasr_official_c64noeqs2max", "ewasr_official_c64noeqs2max_int8")], "ewasr_official"),
]


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"→ paper/figs/{name}.png")


def plot_tradeoff(fps):
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    style(ax)
    for label, col, mk, filled, runs, _ in GROUPS:
        pts = [(fps[h], r["dz"]) for h, n in runs if h in fps and (r := mods(n))]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=42, marker=mk, facecolor=col if filled else "white", edgecolor=col if filled else col,
                   linewidth=1.2, zorder=3, label=label)
    # chỉ chú thích hai điều người đọc cần thấy; danh tính các điểm nằm trong chú giải
    ax.annotate("eWaSR collapses with 2 of 3\ncalibration sets", (18.2, 45), xytext=(20.3, 45), fontsize=7.5,
                color=INK, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate("eWaSR-Lite: 3 calibration sets\nwithin 0.6 points", (29.6, 85.5), xytext=(26.6, 73), fontsize=7.5,
                color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("Accelerator throughput, FPS")
    ax.set_ylabel("Danger-zone F1 after INT8, %")
    ax.set_xlim(14, 32)
    ax.set_ylim(25, 97)
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.3, labelspacing=0.35)
    save(fig, "tradeoff")


def plot_calib():
    keep = [g for g in GROUPS]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7), sharex=True)
    for ax, key, ylab in ((axes[0], "dz", "Danger-zone F1, %"), (axes[1], "dz_pr", "Danger-zone precision, %")):
        style(ax)
        for i, (label, col, mk, filled, runs, fp) in enumerate(keep):
            ref = mods(fp)
            if ref:
                ax.plot([i - 0.28, i + 0.28], [ref[key]] * 2, color=INK, lw=1.4, zorder=2)
            vals = [r[key] for _, n in runs if (r := mods(n))]
            offs = [-0.12, 0.0, 0.12][:len(vals)] if len(vals) > 1 else [0.0]
            ax.scatter([i + o for o in offs], vals, s=34, marker=mk, facecolor=col if filled else "white",
                       edgecolor=col, linewidth=1.2, zorder=3)
        ax.set_ylabel(ylab)
        ax.set_ylim(15, 100)
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels(["E1", "E1\neq. on", "E2", "E6", "E7", "Published\nweights"], fontsize=7.5)
    handles = [Line2D([0], [0], color=INK, lw=1.4, label="FP32"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor=MUTED, markersize=5,
                      label="INT8, one point per calibration set")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=2.0)
    save(fig, "calib")


def plot_gamma():
    rows = [r for r in csv.DictReader(open(ROOT / "docs/data/layer_scale_seeds.csv"))
            if r["khoi"].startswith("mix.") and r["nhanh"] == "γ2"]
    blocks = [int(r["khoi"].split(".")[1]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    style(ax)
    series = [("Published weights", "chinh_thuc", AQUA, "-", "P"), ("E1, run 0", "E1_s0", BLUE, "-", "o"),
              ("E1, run 1", "E1_s1", BLUE, "--", "s"), ("E1, run 2", "E1_s2", BLUE, ":", "^"),
              ("E1, original recipe", "E1_paper", INK, "-.", "D")]
    for label, col_key, col, ls, mk in series:
        if col_key not in rows[0]:
            continue
        ax.plot(blocks, [float(r[col_key]) for r in rows], color=col, ls=ls, lw=1.4, marker=mk, ms=5,
                markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e-42, 1)
    ax.set_yticks([1e-40, 1e-30, 1e-20, 1e-10, 1])
    ax.set_xticks(blocks)
    ax.set_xticklabels([f"{b}" for b in blocks])
    ax.set_xlabel("Feature-mixer block")
    ax.set_ylabel(r"Mean $|\gamma_2|$ of the MLP branch")
    ax.axvspan(0.5, 5.5, color="#f3f3f1", zorder=0)
    ax.text(3.0, 3e-36, "MLPs removed in eWaSR-Lite (blocks 1–5)", ha="center", fontsize=7.5, color=MUTED)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    save(fig, "gamma")


def main():
    fps = hw_fps()
    plot_tradeoff(fps)
    plot_calib()
    plot_gamma()


if __name__ == "__main__":
    main()
