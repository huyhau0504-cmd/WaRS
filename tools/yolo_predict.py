"""Chạy YOLO (Ultralytics, .pt hoặc .onnx) trên USV-VN và lưu box cho tools/evaluate_usv.py.

    python tools/yolo_predict.py --weights runs/yolo/usv_vn/weights/best.pt --split test --out preds/yolov8n
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.data.usv_vn import list_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01, help="ngưỡng thấp; lọc lại khi đánh giá")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    res = {}
    for p in list_images(args.usv_root, args.split):
        r = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        res[p.stem] = [[*map(float, b), float(s)] for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "boxes.json").write_text(json.dumps(res))
    print(f"Đã lưu {out / 'boxes.json'} ({len(res)} ảnh)")


if __name__ == "__main__":
    main()
