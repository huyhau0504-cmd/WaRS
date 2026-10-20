"""ONNX → HAR (parse) → HAR lượng tử hóa INT8 (optimize) → HEF (compile) cho Hailo-8L.

Chạy trong Docker "Hailo AI Software Suite" (có sẵn hailo_sdk_client), mount repo vào container:

    python hailo/compile.py --onnx exports/ewasr_h_384x512.onnx --calib-dir data/mastr1325/images \
        --opt-level 2 --out hailo/build/ewasr_h_384x512_o2

    # Chỉ parse để xem DFC có hỗ trợ hết các lớp không (tuần 3)
    python hailo/compile.py --onnx exports/ewasr_384x512.onnx --parse-only --out hailo/build/ewasr_parse

Chuẩn hóa ImageNet được đưa vào model script (chạy trên chip), nên HEF nhận ảnh RGB uint8.
API theo Hailo Dataflow Compiler 3.x — nếu phiên bản DFC khác, đối chiếu với tài liệu DFC đi kèm.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.constants import MEAN, STD  # noqa: E402


def model_script(opt_level, compression_level, extra, calib_n):
    lines = [
        f"normalization1 = normalization({MEAN.tolist()}, {STD.tolist()})",
        f"model_optimization_flavor(optimization_level={opt_level}, compression_level={compression_level})",
        # Mặc định DFC chỉ dùng 64 ảnh đầu để thu thống kê dù truyền nhiều hơn → đặt rõ số ảnh
        f"model_optimization_config(calibration, calibset_size={calib_n})",
    ]
    if extra:
        lines += Path(extra).read_text().splitlines()
    return "\n".join(lines) + "\n"


def load_calib(calib_dirs, n, hw, seed=0):
    files = sorted(f for d in calib_dirs for f in Path(d).glob("*.jpg"))
    if not files:
        raise SystemExit(f"Không tìm thấy ảnh .jpg trong {calib_dirs}")
    rng = np.random.default_rng(seed)
    files = rng.choice(files, size=min(n, len(files)), replace=False)
    out = np.empty((len(files), hw[0], hw[1], 3), np.float32)
    for i, f in enumerate(files):
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        out[i] = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True, help="tiền tố đầu ra, vd. hailo/build/ewasr_h_o2")
    ap.add_argument("--hw-arch", default="hailo8l")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--calib-dir", nargs="+", default=[], help="một hoặc nhiều thư mục ảnh calibration")
    ap.add_argument("--calib-n", type=int, default=1024)
    ap.add_argument("--opt-level", type=int, default=2, help="0–1 ≈ PTQ; 2–4 bật thêm QFT (cần GPU)")
    ap.add_argument("--compression-level", type=int, default=0)
    ap.add_argument("--alls-extra", help="file chứa thêm lệnh model script (vd. quantization_param 16-bit)")
    ap.add_argument("--from-quantized", action="store_true",
                    help="bỏ qua parse/optimize, biên dịch thẳng từ <out>_quantized.har đã có (khi bước compile bị dừng)")
    args = ap.parse_args()

    from hailo_sdk_client import ClientRunner

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.name.replace("-", "_")

    if args.from_quantized:
        runner = ClientRunner(har=f"{out}_quantized.har", hw_arch=args.hw_arch)
        hef = runner.compile()
        Path(f"{out}.hef").write_bytes(hef)
        print(f"[compile] OK → {out}.hef (từ {out}_quantized.har)")
        return

    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(args.onnx, name)
    runner.save_har(f"{out}.har")
    print(f"[parse] OK → {out}.har")
    if args.parse_only:
        return

    import onnx
    hw = [d.dim_value for d in onnx.load(args.onnx).graph.input[0].type.tensor_type.shape.dim][2:4]
    calib = load_calib(args.calib_dir, args.calib_n, hw)
    script = model_script(args.opt_level, args.compression_level, args.alls_extra, len(calib))
    (out.parent / f"{out.name}.alls").write_text(script)
    runner.load_model_script(script)
    runner.optimize(calib)
    runner.save_har(f"{out}_quantized.har")
    print(f"[optimize] OK → {out}_quantized.har ({len(calib)} ảnh calibration)")

    hef = runner.compile()
    Path(f"{out}.hef").write_bytes(hef)
    print(f"[compile] OK → {out}.hef")


if __name__ == "__main__":
    main()
