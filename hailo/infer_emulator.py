"""Chạy HAR trên bộ giả lập của Hailo DFC (trên laptop) để so FP32 với INT8 mà không cần Pi.

    # INT8 (giả lập lượng tử hóa)
    python hailo/infer_emulator.py --har hailo/build/ewasr_h_o2_quantized.har --ctx quantized \
        --split test --out preds/ewasr_h_int8_o2
    # FP sau các biến đổi của DFC (để tách lỗi do lượng tử hóa với lỗi do biến đổi mô hình)
    python hailo/infer_emulator.py --har hailo/build/ewasr_h_o2_quantized.har --ctx fp_optimized ...

Đầu ra cùng định dạng tools/predict.py → đánh giá bằng tools/evaluate_usv.py.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--ctx", choices=["native", "fp_optimized", "quantized"], default="quantized")
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--split")
    ap.add_argument("--images")
    ap.add_argument("--mods")
    ap.add_argument("--size", type=int, nargs=2, default=[384, 512], help="H W đầu vào của mô hình")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    from hailo_sdk_client import ClientRunner, InferenceContext
    ctx_type = {"native": InferenceContext.SDK_NATIVE, "fp_optimized": InferenceContext.SDK_FP_OPTIMIZED,
                "quantized": InferenceContext.SDK_QUANTIZED}[args.ctx]
    runner = ClientRunner(har=args.har)
    hw = tuple(args.size)

    if args.mods:
        items = list_mods_frames(args.mods)
    else:
        src = list_images(args.usv_root, args.split) if args.split else sorted(Path(args.images).glob("*.jpg"))
        items = [(None, p) for p in src]

    out = Path(args.out)
    with runner.infer_context(ctx_type) as ctx:
        for i in tqdm(range(0, len(items), args.batch)):
            chunk = items[i:i + args.batch]
            imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for _, p in chunk]
            # SDK_NATIVE chưa có lớp normalization của model script → tự chuẩn hóa
            x = np.stack([cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for im in imgs])
            x = x.astype(np.float32)
            if args.ctx == "native":
                x = (x - MEAN) / STD
            y = runner.infer(ctx, x)  # NHWC
            for (seq, p), img, logits in zip(chunk, imgs, y):
                logits = logits.transpose(2, 0, 1)
                if seq is not None:
                    mask = logits_to_mask(logits, hw)
                    save_mask(cv2.cvtColor(MODS_COLORS[mask], cv2.COLOR_RGB2BGR), out / seq / f"{p.stem}.png")
                    continue
                mask = logits_to_mask(logits, img.shape[:2])
                save_mask(mask, out / "masks" / f"{p.stem}.png")
                if args.vis:
                    save_mask(cv2.cvtColor(overlay(img, mask), cv2.COLOR_RGB2BGR), out / "vis" / f"{p.stem}.jpg")


if __name__ == "__main__":
    main()
