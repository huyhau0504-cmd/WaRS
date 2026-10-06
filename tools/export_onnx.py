"""Xuất mô hình ra ONNX (shape tĩnh) để đưa vào Hailo DFC, và kiểm tra sai số so với PyTorch.

    python tools/export_onnx.py --model ewasr_h --weights runs/ewasr_h/best.pth --size 384 512
    → exports/ewasr_h_384x512.onnx  (input "image" 1x3xHxW đã chuẩn hóa, output "logits" 1x3xH/4xW/4)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.models import get_model, load_weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--size", type=int, nargs=2, default=[384, 512], help="H W, chia hết cho 32")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--out")
    args = ap.parse_args()
    h, w = args.size
    assert h % 32 == 0 and w % 32 == 0, "H, W phải chia hết cho 32"

    model = get_model(args.model, pretrained_backbone=False)
    load_weights(model, args.weights)
    model.eval()

    out = Path(args.out or f"exports/{args.model}_{h}x{w}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, h, w)
    torch.onnx.export(model, (x,), str(out), opset_version=args.opset, input_names=["image"],
                      output_names=["logits"], dynamo=False)

    try:
        import onnx
        import onnxsim
        m, ok = onnxsim.simplify(onnx.load(str(out)))
        if ok:
            onnx.save(m, str(out))
    except ImportError:
        print("(bỏ qua onnxsim — chưa cài)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out))
    y_onnx = sess.run(None, {"image": x.numpy()})[0]
    with torch.no_grad():
        y = model(x).numpy()
    print(f"Đã xuất {out}  output {y_onnx.shape}  max|torch-onnx|={np.abs(y - y_onnx).max():.2e}")


if __name__ == "__main__":
    main()
