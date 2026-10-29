"""So khoảng lượng tử đầu ra (limvals_out) theo lớp giữa hai HAR đã lượng tử hóa của cùng một mô hình (Q-012).

    python hailo/limvals_compare.py --pair E1=hailo/build/e1_ewasr_o0_quantized.har,hailo/build/e1_ewasr_c1024max_quantized.har \\
        --out docs/data/limvals_growth.csv

Mỗi dòng: run, lớp Hailo, tên gốc ONNX (2 tên đầu), khoảng [lo, hi] của HAR A và B, growth = độ rộng B / độ rộng A.
Chạy trong venv DFC. Nạp lần lượt từng HAR để giữ RAM thấp (HAR eWaSR đầy đủ ~1,2 GB).
"""
import argparse
import csv
import gc

import numpy as np


def limvals(har):
    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=har, hw_arch="hailo8l")
    names = {l.name.split("/")[-1]: ",".join(getattr(l, "original_names", [])[:2])
             for l in runner.get_hn_model().stable_toposort()}
    out = {}
    for k, v in runner.get_params_translated().items():
        if k.endswith("/limvals_out:0"):
            layer = k.split("/")[1]
            v = np.asarray(v, np.float64).ravel()
            out[layer] = (float(v[0]), float(v[-1]), names.get(layer, ""))
    del runner
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs="+", required=True, help="TÊN=har_A,har_B")
    ap.add_argument("--out", default="docs/data/limvals_growth.csv")
    args = ap.parse_args()
    rows = []
    for spec in args.pair:
        tag, hars = spec.split("=", 1)
        har_a, har_b = hars.split(",")
        a, b = limvals(har_a), limvals(har_b)
        for layer, (lo_a, hi_a, name) in a.items():
            if layer not in b:
                continue
            lo_b, hi_b, _ = b[layer]
            growth = (hi_b - lo_b) / max(hi_a - lo_a, 1e-30)
            rows.append([tag, layer, name or b[layer][2], f"{lo_a:.4g}", f"{hi_a:.4g}", f"{lo_b:.4g}", f"{hi_b:.4g}",
                         f"{growth:.4g}"])
        g = [float(r[-1]) for r in rows if r[0] == tag]
        print(f"[{tag}] {len(g)} lớp · growth max {max(g):.2f} · số lớp > 1,5×: {sum(x > 1.5 for x in g)}", flush=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "layer", "onnx_names", "lo_a", "hi_a", "lo_b", "hi_b", "growth"])
        w.writerows(rows)
    print(f"→ {args.out} ({len(rows)} dòng)")


if __name__ == "__main__":
    main()
