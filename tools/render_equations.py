"""Vẽ phương trình LaTeX (mathtext của matplotlib, font STIX giống Times) thành PNG cho bản Word của bài báo.

    third_party/.venv-mods/bin/python tools/render_equations.py eqs.json out_dir

eqs.json: [{"name": "mf1", "tex": "x \\leftarrow ..."}, ...] → out_dir/<name>.png (600 dpi, nền trong suốt).
Dùng bởi tools/build_eej_docx.py; tạp chí yêu cầu MathType nên ảnh chỉ là bản tạm để duyệt nội dung.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})


def main():
    eqs = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for e in eqs:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${e['tex']}$", fontsize=12)
        fig.savefig(out / f"{e['name']}.png", dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
        print(e["name"])


if __name__ == "__main__":
    main()
