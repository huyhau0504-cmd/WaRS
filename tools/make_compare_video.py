"""Video demo so sánh eWaSR-Lite với các mô hình thị giác–ngôn ngữ trên một sequence MODS (lưới 2×2).

    python tools/make_compare_video.py --seq kope103-00036460-00039500 \
        --mask-dir preds/mods/e7_ewasr_h_lite_c64noeqmax_int8 \
        --boxes "PaliGemma 2=~/Downloads/results_V4/runs/vlm_paligemma2/boxes.json" \
                "Qwen2.5-VL-3B=~/Downloads/results_V4/runs/vlm_qwen25vl/boxes.json" \
                "OWLv2=~/Downloads/results_V5/runs/vlm_owlv2/boxes.json:0.2" \
        --out results/demo/compare_vlm.mp4 [--realtime]

Ô 1: eWaSR-Lite (mask INT8 do Hailo-8L ghi, vật cản vàng / nước xanh). Các ô còn lại: khung bao của từng VLM
(":ngưỡng" sau đường dẫn để lọc điểm, cho OWLv2). Mỗi ô ghi thời gian xử lý một ảnh (trung vị trong boxes.json;
eWaSR-Lite: --ewasr-ms). Mặc định mọi ô dùng kết quả của chính khung đó (so độ chính xác). Với --realtime, video
chạy theo thời gian thực của chuỗi MODS (khung trái cách nhau --frame-dt giây): một mô hình chỉ bắt đầu ảnh mới khi
xong ảnh trước, nên ô của nó giữ kết quả cũ (khung bao cũ vẽ lên ảnh hiện tại) — mô phỏng độ trễ khi thuyền đang
chạy, với thời gian đo trên GPU T4 (VLM) và trên Pi 5 + Hailo-8L (eWaSR-Lite).
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
BOX_COLORS = [(52, 104, 235), (122, 175, 27), (214, 120, 42)]  # BGR: cam, xanh ngọc, xanh dương


def read_mods_mask(path):
    """Mask màu MODS (3 kênh) hoặc mask mã lớp 0/1/2 (1 kênh, như preds/lars/*/masks) → mã lớp."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 2:
        return raw
    rgb = raw[:, :, ::-1]
    cls = np.full(rgb.shape[:2], 255, np.uint8)
    for c, col in enumerate(MODS_COLORS):
        cls[(rgb == col).all(-1)] = c
    return cls


def overlay(img, cls, alpha=0.45):
    cls = cv2.resize(cls, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    for c in (0, 1):  # chỉ tô vật cản và nước; trời để nguyên cho dễ nhìn
        m = cls == c
        out[m] = (img[m] * (1 - alpha) + VIS_COLORS[c][::-1] * alpha).astype(np.uint8)
    return out


def draw_boxes(img, boxes, color, thr):
    out = img.copy()
    n = 0
    for x1, y1, x2, y2, score, label in boxes:
        if score < thr:
            continue
        n += 1
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 3)
        if label:
            cv2.putText(out, str(label)[:16], (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out, n


def banner(img, title, sub, stale=None):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, sub, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
    if stale:
        cv2.putText(out, stale, (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods-root", default=str(Path.home() / "datasets/mods"))
    ap.add_argument("--seq", help="MODS: tên sequence")
    ap.add_argument("--images-dir", help="hoặc ảnh rời (LaRS val/images): trình chiếu, mỗi ảnh giữ --hold khung")
    ap.add_argument("--scene", choices=["river_like", "sea_like"], help="LaRS: chỉ lấy ảnh có scene_type này")
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--hold", type=int, default=20, help="số khung video cho mỗi ảnh rời")
    ap.add_argument("--mask-dir", default=str(ROOT / "preds/mods/e7_ewasr_h_lite_c64noeqmax_int8"))
    ap.add_argument("--ewasr-ms", type=float, default=44.0, help="thời gian/ảnh của eWaSR-Lite (đầu-cuối trên Pi 5)")
    ap.add_argument("--ewasr-name", default="eWaSR-Lite (ours)")
    ap.add_argument("--boxes", nargs="+", required=True, help="'Tên=đường/dẫn/boxes.json[:ngưỡng]'")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--frame-dt", type=float, default=1.0,
                    help="giây giữa hai khung trái liên tiếp của MODS (khung có nhãn cách nhau 10 chỉ số; giả định luồng 10 Hz)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--tile", type=int, nargs=2, default=[640, 480], help="W H mỗi ô")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.images_dir:
        imgs = Path(args.images_dir).expanduser()
        frames = sorted(p for p in imgs.iterdir() if p.suffix.lower() in (".jpg", ".png"))
        if args.scene:
            ann = json.loads((imgs.parent / "image_annotations.json").read_text())["annotations"]
            keep = {Path(a["file_name"]).stem for a in ann if a["labels"]["scene_type"] == args.scene}
            frames = [f for f in frames if f.stem in keep]
        frames = frames[:args.max_images]
        key = lambda f: f.stem  # noqa: E731
        mask_path = lambda f: Path(args.mask_dir) / f"{f.stem}.png"  # noqa: E731
    else:
        frames = sorted((Path(args.mods_root).expanduser() / "sequences" / args.seq / "frames").glob("*L.jpg"))
        key = lambda f: f"{args.seq}/{f.stem}"  # noqa: E731
        mask_path = lambda f: Path(args.mask_dir) / args.seq / f"{f.stem}.png"  # noqa: E731
    models = []
    for k, spec in enumerate(args.boxes):
        name, rest = spec.split("=", 1)
        path, _, thr = rest.partition(":") if rest.count(":") else (rest, "", "")
        res = json.loads(Path(path).expanduser().read_text())
        times = res["meta"].get("seconds_per_image", [])
        med = sorted(times)[len(times) // 2] if times else float("nan")
        models.append({"name": name, "frames": res["frames"], "thr": float(thr or 0), "sec": med,
                       "color": BOX_COLORS[k % len(BOX_COLORS)], "device": res["meta"].get("device", "")})

    W, H = args.tile
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * W, 2 * H))
    # thời gian thực: mỗi mô hình xử lý xong khung bắt đầu ở t0 tại t0 + sec; khi rảnh lấy khung mới nhất
    # mô hình rảnh thì nhận khung hiện tại; kết quả dùng được nếu xong trước khi khung kế tiếp tới
    state = [{"busy_until": 0.0, "done": []} for _ in range(1 + len(models))]
    secs = [args.ewasr_ms / 1000] + [m["sec"] for m in models]
    for i, f in enumerate(frames):
        t = i * args.frame_dt
        img = cv2.imread(str(f))
        src = [i] * (1 + len(models))
        if args.realtime:
            for j, st in enumerate(state):
                if t >= st["busy_until"] - 1e-9:
                    st["busy_until"] = t + secs[j]
                    st["done"].append((st["busy_until"], i))
                ready = [k for fin, k in st["done"] if fin <= t + args.frame_dt]
                src[j] = ready[-1] if ready else None
        tiles = []
        # ô 1: eWaSR-Lite
        j = src[0]
        sub = f"Pi 5 + Hailo-8L INT8 · {args.ewasr_ms / 1000:.3f} s/anh"
        if j is None:
            tile = banner(img, args.ewasr_name, sub, "dang xu ly anh dau...")
        else:
            m = read_mods_mask(mask_path(frames[j]))
            tile = banner(overlay(img, m) if m is not None else img, args.ewasr_name, sub,
                          f"ket qua cua {(i - j) * args.frame_dt:.1f} s truoc" if i - j > 0 else None)
        tiles.append(tile)
        for k, m in enumerate(models):
            j = src[k + 1]
            sub = f"{m['device'].replace('Tesla ', '')} · {m['sec']:.2f} s/anh"
            if j is None:
                tiles.append(banner(img, m["name"], sub, "dang xu ly anh dau..."))
                continue
            boxes = m["frames"].get(key(frames[j]), [])
            drawn, _ = draw_boxes(img, boxes, m["color"], m["thr"])
            tiles.append(banner(drawn, m["name"], sub,
                                f"khung bao cua {(i - j) * args.frame_dt:.1f} s truoc" if i - j > 0 else None))
        while len(tiles) < 4:
            tiles.append(np.zeros_like(img))
        tiles = [cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA) for x in tiles[:4]]
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        for _ in range(args.hold if args.images_dir else 1):
            vw.write(grid)
    vw.release()
    print(f"{len(frames)} khung → {out}" + (" (thời gian thực)" if args.realtime else ""))


if __name__ == "__main__":
    main()
