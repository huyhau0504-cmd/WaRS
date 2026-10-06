"""Chạy mô hình phân đoạn (PyTorch .pth hoặc ONNX) và lưu mask lớp.

    # USV-VN, PyTorch
    python tools/predict.py --model ewasr --weights weights/ewasr_resnet18.pth --split test --out preds/ewasr_fp32
    # ONNX (kiểm tra file xuất ra trước khi đưa vào Hailo)
    python tools/predict.py --onnx exports/ewasr_h_384x512.onnx --split test --out preds/ewasr_h_onnx
    # MODS: ghi mask màu theo từng sequence cho MODS evaluation toolkit
    python tools/predict.py --model ewasr --weights ... --mods /path/to/mods --out preds/mods/ewasr

Đầu ra (USV-VN / --images): <out>/masks/<stem>.png (mã lớp 0/1/2, cùng kích thước ảnh gốc),
tùy chọn <out>/vis/<stem>.jpg. Đánh giá bằng tools/evaluate_usv.py.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.data.mods import list_mods_frames  # noqa: E402
from wars.constants import MEAN, STD  # noqa: E402
from wars.data.usv_vn import list_images  # noqa: E402
from wars.postprocess import MODS_COLORS, logits_to_mask, overlay, save_mask  # noqa: E402


class TorchRunner:
    def __init__(self, name, weights, size):
        import torch
        from wars.models import get_model, load_weights
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(name, pretrained_backbone=False)
        load_weights(self.model, weights)
        self.model.eval().to(self.device)
        self.size = size

    def __call__(self, x):
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x).to(self.device)).cpu().numpy()


class OnnxRunner:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=ort.get_available_providers())
        self.inp = self.sess.get_inputs()[0]
        self.size = tuple(self.inp.shape[2:4])

    def __call__(self, x):
        return self.sess.run(None, {self.inp.name: x})[0]


def preprocess(img_rgb, size):
    x = cv2.resize(img_rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="tên mô hình (wars.models.MODELS), dùng với --weights")
    ap.add_argument("--weights")
    ap.add_argument("--onnx")
    ap.add_argument("--size", type=int, nargs=2, default=[384, 512], help="H W (chỉ cho PyTorch)")
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--split", help="train | valid | test của USV-VN")
    ap.add_argument("--images", help="thư mục ảnh bất kỳ")
    ap.add_argument("--mods", help="thư mục gốc MODS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vis", action="store_true", help="lưu ảnh overlay")
    args = ap.parse_args()

    run = OnnxRunner(args.onnx) if args.onnx else TorchRunner(args.model, args.weights, tuple(args.size))
    out = Path(args.out)

    if args.mods:
        items = [(seq, p) for seq, p in list_mods_frames(args.mods)]
    else:
        src = list_images(args.usv_root, args.split) if args.split else sorted(Path(args.images).glob("*.jpg"))
        items = [(None, p) for p in src]

    for seq, p in tqdm(items):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        logits = run(preprocess(img, run.size))[0]
        if seq is not None:
            # MODS: lưu ở kích thước đầu vào mô hình, như WaSR predict_mods.py
            mask = logits_to_mask(logits, run.size)
            save_mask(cv2.cvtColor(MODS_COLORS[mask], cv2.COLOR_RGB2BGR), out / seq / f"{p.stem}.png")
            continue
        mask = logits_to_mask(logits, img.shape[:2])
        save_mask(mask, out / "masks" / f"{p.stem}.png")
        if args.vis:
            save_mask(cv2.cvtColor(overlay(img, mask), cv2.COLOR_RGB2BGR), out / "vis" / f"{p.stem}.jpg")


if __name__ == "__main__":
    main()
