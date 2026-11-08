"""Dựng bản Word của bài báo theo mẫu AUT Journal of Electrical Engineering (EEJ) từ paper/eej/manuscript.md.

    third_party/.venv-docx/bin/python tools/build_eej_docx.py
    → paper/eej/WaRS_EEJ_manuscript.docx

Cần python-docx (venv third_party/.venv-docx) và matplotlib để vẽ phương trình (gọi
third_party/.venv-mods/bin/python tools/render_equations.py). File mẫu: --template (mặc định file .docx của EEJ trong
~/Downloads). Giữ style của mẫu (Times, giãn dòng đôi, căn đều, chú thích bảng/hình theo style của tạp chí), đổi khổ
giấy sang A4, đánh số mục kiểu "1-", "1-1-", trích dẫn [n] theo thứ tự xuất hiện, bảng có chú thích ở trên, hình có
chú thích ở dưới. Phương trình là ảnh tạm: tạp chí yêu cầu gõ lại bằng MathType trước khi nộp.

Cú pháp manuscript.md: xem đầu file đó.
"""
import argparse
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips

ROOT = Path(__file__).resolve().parents[1]
FONT = "Times New Roman"
BODY_PT = 11
CAPTION_STYLE = "بالانویس جدول انگلیسی"  # style chú thích bảng/hình của mẫu EEJ
TEXT_W = Inches(6.27)  # A4 trừ lề 1 inch

INLINE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|\^(.+?)\^|~(.+?)~|\[(@[^\]]+)\]|@(tab|fig|eq):([\w-]+)|\[TBD:([^\]]*)\]")


# ---------------------------------------------------------------- đọc manuscript

def parse(path):
    meta, blocks, refs, nom = {"affil": []}, [], {}, []
    lines = path.read_text().splitlines()
    i = 0
    para = []

    def flush():
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("%%"):
            i += 1
            continue
        m = re.match(r"%(\w+): ?(.*)", ln)
        if m and m.group(1) in ("title", "authors", "corresponding", "keywords", "abstract"):
            meta[m.group(1)] = m.group(2)
        elif m and m.group(1) == "affil":
            n, t = m.group(2).split("|", 1)
            meta["affil"].append((n.strip(), t.strip()))
        elif ln.startswith("%ref "):
            k, t = ln[5:].split("|", 1)
            refs[k.strip()] = t.strip()
        elif ln.startswith("%nomhead "):
            nom.append(("head", ln[9:].strip(), ""))
        elif ln.startswith("%nom "):
            s, t = ln[5:].split("|", 1)
            nom.append(("row", s.strip(), t.strip()))
        elif not ln.strip():
            flush()
        elif re.match(r"#{1,3} ", ln):
            flush()
            level = len(ln.split(" ")[0])
            text = ln[level + 1:]
            numbered = not text.endswith("{-}")
            blocks.append(("h", level, text.replace("{-}", "").strip(), numbered))
        elif ln.startswith("$$"):
            flush()
            m = re.match(r"\$\$(.+)\$\$\s*\{#eq:([\w-]+)\}", ln)
            blocks.append(("eq", m.group(2), m.group(1).strip()))
        elif ln.startswith("Table: "):
            flush()
            m = re.match(r"Table: (.+?)\s*\{#tab:([\w-]+)\}", ln)
            rows = []
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", m.group(2), m.group(1), rows))
            continue
        elif ln.startswith("!["):
            flush()
            m = re.match(r"!\[(.+)\]\((.+?)\)\{#fig:([\w-]+)\}", ln)
            blocks.append(("fig", m.group(3), m.group(1), (path.parent / m.group(2)).resolve()))
        elif re.match(r"\d+\. ", ln):
            flush()
            blocks.append(("li", ln))
        else:
            para.append(ln.strip())
        i += 1
    flush()
    return meta, blocks, refs, nom


def number_things(blocks):
    """Số bảng/hình/phương trình theo thứ tự định nghĩa; trích dẫn theo thứ tự xuất hiện."""
    nums = {"tab": {}, "fig": {}, "eq": {}}
    cites = {}
    for b in blocks:
        if b[0] in ("table", "fig", "eq"):
            kind = {"table": "tab", "fig": "fig", "eq": "eq"}[b[0]]
            nums[kind][b[1]] = len(nums[kind]) + 1
    texts = []
    for b in blocks:
        if b[0] in ("p", "li"):
            texts.append(b[1])
        elif b[0] == "table":
            texts.append(b[2])
            texts.extend(" ".join(r) for r in b[3])
        elif b[0] == "fig":
            texts.append(b[2])
    for t in texts:
        for g in re.findall(r"\[(@[^\]]+)\]", t):
            for k in re.findall(r"@([\w-]+)", g):
                cites.setdefault(k, len(cites) + 1)
    return nums, cites


def cite_label(keys, cites):
    ns = sorted(cites[k] for k in keys)
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"{ns[i]}-{ns[j]}" if j - i >= 2 else ", ".join(map(str, ns[i:j + 1])))
        i = j + 1
    return "[" + ", ".join(out) + "]"


# ---------------------------------------------------------------- viết docx

def set_font(run, size=BODY_PT, bold=None, italic=None):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), FONT)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_inline(p, text, ctx, size=BODY_PT, bold=False, italic=False):
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            set_font(p.add_run(text[pos:m.start()]), size, bold, italic)
        b, it, sup, sub, cite, kind, name, tbd = m.groups()
        if b is not None:
            add_inline(p, b, ctx, size, True, italic)
        elif it is not None:
            add_inline(p, it, ctx, size, bold, True)
        elif sup is not None:
            r = p.add_run(sup)
            set_font(r, size, bold, italic)
            r.font.superscript = True
        elif sub is not None:
            r = p.add_run(sub)
            set_font(r, size, bold, italic)
            r.font.subscript = True
        elif cite is not None:
            keys = re.findall(r"@([\w-]+)", cite)
            missing = [k for k in keys if k not in ctx["refs"]]
            if missing:
                raise SystemExit(f"thiếu %ref cho {missing}")
            set_font(p.add_run(cite_label(keys, ctx["cites"])), size, bold, italic)
        elif kind is not None:
            n = ctx["nums"][kind].get(name)
            if n is None:
                raise SystemExit(f"tham chiếu không tồn tại: @{kind}:{name}")
            set_font(p.add_run(str(n)), size, bold, italic)
        elif tbd is not None:
            r = p.add_run(f"[TBD:{tbd}]")
            set_font(r, size, bold, italic)
            r.font.color.rgb = RGBColor(0xC0, 0, 0)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        pos = m.end()
    if pos < len(text):
        set_font(p.add_run(text[pos:]), size, bold, italic)


def body_par(d, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=2.0, before=0, after=6, keep=False):
    p = d.add_paragraph()
    pf = p.paragraph_format
    p.alignment = align
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if keep:
        pf.keep_with_next = True
    return p


def caption(d, label, text, ctx, keep_next):
    p = d.add_paragraph(style=CAPTION_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = keep_next
    set_font(p.add_run(label + " "), 10, True)
    add_inline(p, text, ctx, 10)


def border(el, edges, sz=8):
    tcpr = el.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for e in edges:
        x = OxmlElement(f"w:{e}")
        x.set(qn("w:val"), "single")
        x.set(qn("w:sz"), str(sz))
        x.set(qn("w:color"), "000000")
        b.append(x)
    tcpr.append(b)


def add_table(d, rows, ctx):
    notes = []
    while rows and rows[-1][0][:1] in ("*", "†") and not any(c for c in rows[-1][1:]):
        notes.insert(0, rows.pop()[0])
    ncol = len(rows[0])
    t = d.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(rows):
        for c in range(ncol):
            cell = t.cell(r, c)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            add_inline(p, row[c] if c < len(row) else "", ctx, 9.5, bold=(r == 0))
            edges = []
            if r == 0:
                edges += ["top", "bottom"]
            if r == len(rows) - 1:
                edges.append("bottom")
            if edges:
                border(cell._tc, edges)
    for n in notes:
        p = body_par(d, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 2, 2)
        add_inline(p, n, ctx, 9)
    body_par(d, spacing=1.0, after=0)  # một dòng trống sau bảng, như mẫu


def render_equations(blocks, tmp):
    eqs = [{"name": b[1], "tex": b[2]} for b in blocks if b[0] == "eq"]
    (tmp / "eqs.json").write_text(json.dumps(eqs))
    subprocess.run([str(ROOT / "third_party/.venv-mods/bin/python"), str(ROOT / "tools/render_equations.py"),
                    str(tmp / "eqs.json"), str(tmp)], check=True, capture_output=True)


def add_footer_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            f = OxmlElement("w:fldChar")
            f.set(qn("w:fldCharType"), kind)
            r._element.append(f)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            r._element.append(it)
    set_font(r, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "paper/eej/manuscript.md"))
    ap.add_argument("--template", default=str(Path.home() / "Downloads/AUT_Journal_English_Full_Paper_Template_v7.2.28_(EEJ).docx"))
    ap.add_argument("--out", default=str(ROOT / "paper/eej/WaRS_EEJ_manuscript.docx"))
    args = ap.parse_args()

    meta, blocks, refs, nom = parse(Path(args.src))
    nums, cites = number_things(blocks)
    unused = [k for k in refs if k not in cites]
    ctx = {"nums": nums, "cites": cites, "refs": refs}

    d = docx.Document(args.template)
    body = d.element.body
    for el in list(body):
        if el.tag != qn("w:sectPr"):
            body.remove(el)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Twips(11906), Twips(16838)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    add_footer_page_number(sec)

    # trang đầu: tiêu đề, tác giả, đơn vị, tóm tắt, từ khóa
    p = body_par(d, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 0, 12)
    add_inline(p, meta["title"], ctx, 14, bold=True)
    p = body_par(d, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 0, 6)
    add_inline(p, meta["authors"], ctx)
    for n, t in meta["affil"]:
        p = body_par(d, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 0, 0)
        add_inline(p, f"^{n}^{t}", ctx, 10)
    p = body_par(d, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 6, 12)
    add_inline(p, f"*Corresponding author's email: {meta['corresponding']}", ctx, 10)
    p = body_par(d, after=0)
    set_font(p.add_run("Abstract:"), bold=True)
    p = body_par(d)
    add_inline(p, meta["abstract"], ctx)
    p = body_par(d, after=0)
    set_font(p.add_run("Keywords:"), bold=True)
    p = body_par(d)
    add_inline(p, meta["keywords"], ctx)
    p.add_run().add_break(WD_BREAK.PAGE)

    counters = [0, 0, 0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        render_equations(blocks, tmp)
        for k, b in enumerate(blocks):
            nxt = blocks[k + 1] if k + 1 < len(blocks) else None
            if b[0] == "h":
                _, level, text, numbered = b
                label = ""
                if numbered:
                    counters[level - 1] += 1
                    for j in range(level, 3):
                        counters[j] = 0
                    label = "-".join(str(c) for c in counters[:level]) + "- "
                p = body_par(d, WD_ALIGN_PARAGRAPH.LEFT, 2.0, 12 if level == 1 else 6, 0, keep=True)
                add_inline(p, label + text, ctx, BODY_PT + (1 if level == 1 else 0), bold=True)
                if text == "References":
                    for key, n in sorted(cites.items(), key=lambda kv: kv[1]):
                        rp = body_par(d, spacing=1.0, after=4)
                        rp.paragraph_format.left_indent = Inches(0.4)
                        rp.paragraph_format.first_line_indent = Inches(-0.4)
                        add_inline(rp, f"[{n}]\t{refs[key]}", ctx, 10)
                        rp.paragraph_format.tab_stops.add_tab_stop(Inches(0.4))
                if text == "Nomenclature":
                    for kind, s, t in nom:
                        np_ = body_par(d, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 4 if kind == "head" else 0, 2)
                        if kind == "head":
                            add_inline(np_, s, ctx, 10, bold=True, italic=True)
                        else:
                            np_.paragraph_format.tab_stops.add_tab_stop(Inches(1.0))
                            add_inline(np_, s, ctx, 10, italic=s not in ("FPS", "GMAC", "HFOV", "INT8", "MLP", "PTQ", "USV"))
                            add_inline(np_, "\t" + t, ctx, 10)
            elif b[0] == "p":
                add_inline(body_par(d), b[1], ctx)
            elif b[0] == "li":
                p = body_par(d)
                p.paragraph_format.left_indent = Inches(0.3)
                p.paragraph_format.first_line_indent = Inches(-0.3)
                add_inline(p, re.sub(r"^(\d+)\. ", r"\1.\t", b[1]), ctx)
                p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
            elif b[0] == "eq":
                p = body_par(d, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 6, 6)
                ts = p.paragraph_format.tab_stops
                ts.add_tab_stop(TEXT_W // 2, WD_TAB_ALIGNMENT.CENTER)
                ts.add_tab_stop(TEXT_W, WD_TAB_ALIGNMENT.RIGHT)
                p.add_run("\t")
                img = tmp / f"{b[1]}.png"
                w_in = struct.unpack(">I", img.read_bytes()[16:20])[0] / 600  # bề rộng PNG ở 600 dpi → giữ cỡ chữ 12 pt
                p.add_run().add_picture(str(img), width=Inches(w_in))
                set_font(p.add_run(f"\t({nums['eq'][b[1]]})"))
            elif b[0] == "table":
                caption(d, f"Table {nums['tab'][b[1]]}.", b[2], ctx, True)
                add_table(d, [list(r) for r in b[3]], ctx)
            elif b[0] == "fig":
                p = body_par(d, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 6, 0, keep=True)
                p.add_run().add_picture(str(b[3]), width=Inches(6.2))
                caption(d, f"Fig. {nums['fig'][b[1]]}.", b[2], ctx, False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.save(args.out)
    print(f"→ {args.out}: {len(cites)} tài liệu tham khảo, {len(nums['tab'])} bảng, {len(nums['fig'])} hình, "
          f"{len(nums['eq'])} phương trình" + (f"; %ref không dùng: {unused}" if unused else ""))


if __name__ == "__main__":
    main()
