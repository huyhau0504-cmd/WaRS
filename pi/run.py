"""Pipeline nhận diện vật cản trên Raspberry Pi 5 + Hailo-8L (HailoRT).

    # Phát lại tập test USV-VN, lưu mask để đánh giá + đo thời gian từng bước
    python3 pi/run.py --hef ewasr_h_384x512.hef --split test --out results/ewasr_h_pi --hfov 70
    # Video / camera
    python3 pi/run.py --hef ewasr_h_384x512.hef --video 0 --show

Mặc định chạy pipeline 3 luồng (đọc ảnh → suy luận Hailo → hậu xử lý CPU) để hậu xử lý khung n−1 chồng
lên suy luận khung n; `fps_throughput` là thông lượng thật. `--sequential` chạy tuần tự như cũ (đo độ trễ
từng bước không bị tranh chấp).

Trên Pi chỉ cần: numpy, opencv-python và HailoRT Python (`sudo apt install hailo-all`).
FPS thuần của HEF (không tính tiền/hậu xử lý): `hailortcli benchmark <file.hef>`.
"""
import argparse
import json
import multiprocessing as mp
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.data.usv_vn import list_images  # noqa: E402
from wars.postprocess import extract_obstacles, logits_to_mask, overlay, save_mask  # noqa: E402


class HailoSeg:
    def __init__(self, hef_path):
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams, VDevice)
        self.hef = HEF(hef_path)
        self.device = VDevice()
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.device.configure(self.hef, cfg)[0]
        self.ng_params = self.ng.create_params()
        inp = self.hef.get_input_vstream_infos()[0]
        self.in_name, self.hw = inp.name, tuple(inp.shape[:2])
        self.out_name = self.hef.get_output_vstream_infos()[0].name
        self.pipe = InferVStreams(self.ng, InputVStreamParams.make(self.ng, format_type=FormatType.UINT8),
                                  OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32))
        self.pipe.__enter__()
        self.act = self.ng.activate(self.ng_params)
        self.act.__enter__()

    def __call__(self, img_rgb):
        x = cv2.resize(img_rgb, (self.hw[1], self.hw[0]), interpolation=cv2.INTER_AREA)[None]
        y = self.pipe.infer({self.in_name: x})[self.out_name][0]  # HWC
        return y.transpose(2, 0, 1)

    def close(self):
        self.act.__exit__(None, None, None)
        self.pipe.__exit__(None, None, None)
        self.device.release()


def post_process(logits, hw, min_area, hfov):
    """Hậu xử lý một khung (chạy trong tiến trình con ở chế độ pipeline)."""
    mask = logits_to_mask(logits, hw)
    return mask, extract_obstacles(mask, min_area, hfov)


def frames(args):
    if args.video is not None:
        cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f"frame_{i:06d}", f
            i += 1
    else:
        src = list_images(args.usv_root, args.split) if args.split else sorted(Path(args.images).glob("*.jpg"))
        for p in src:
            yield p.stem, cv2.imread(str(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--usv-root", default="data/usv_vn")
    ap.add_argument("--split")
    ap.add_argument("--images")
    ap.add_argument("--video", help="đường dẫn video hoặc chỉ số camera")
    ap.add_argument("--out")
    ap.add_argument("--hfov", type=float, help="góc nhìn ngang camera (độ) để tính hướng vật cản")
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--sequential", action="store_true", help="tắt pipeline, chạy tuần tự từng khung")
    args = ap.parse_args()

    seg = HailoSeg(args.hef)
    out = Path(args.out) if args.out else None
    times = {"pre+infer": [], "post": [], "total": []}
    log = None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        log = (out / "obstacles.jsonl").open("w")
    def handle(name, img, mask, obstacles):
        if log:
            log.write(json.dumps({"frame": name, "obstacles": obstacles}) + "\n")
            save_mask(mask, out / "masks" / f"{name}.png")
        if args.vis or args.show:
            vis = cv2.cvtColor(overlay(img, mask), cv2.COLOR_RGB2BGR)
            for ob in obstacles:
                x1, y1, x2, y2 = ob["bbox"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if args.vis and out:
                save_mask(vis, out / "vis" / f"{name}.jpg")
            if args.show:
                cv2.imshow("WaRS", vis)
                return cv2.waitKey(1) != ord("q")
        return True

    def post(img, logits):
        return post_process(logits, img.shape[:2], args.min_area, args.hfov)

    wall0 = time.perf_counter()
    try:
        if args.sequential:
            for name, bgr in frames(args):
                t0 = time.perf_counter()
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                logits = seg(img)
                t1 = time.perf_counter()
                mask, obstacles = post(img, logits)
                t2 = time.perf_counter()
                times["pre+infer"].append(t1 - t0); times["post"].append(t2 - t1); times["total"].append(t2 - t0)
                if not handle(name, img, mask, obstacles):
                    break
        else:
            # HailoRT Python giữ GIL trong lúc chờ chip, nên hậu xử lý phải chạy ở tiến trình khác mới chồng
            # được lên suy luận. Pool tạo bằng "spawn" để tiến trình con không thừa hưởng handle của thiết bị.
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"))
            pool.submit(post_process, np.zeros((3, 8, 8), np.float32), (32, 32), 1, None).result()  # khởi động trước
            stop, q_in, q_post = object(), queue.Queue(4), queue.Queue(4)
            wall0 = time.perf_counter()

            def reader():
                for name, bgr in frames(args):
                    q_in.put((name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                q_in.put(stop)

            def infer():
                while (it := q_in.get()) is not stop:
                    name, img = it
                    t0 = time.perf_counter()  # độ trễ tính từ lúc bắt đầu suy luận, không gồm thời gian chờ hàng đợi
                    logits = seg(img)
                    t1 = time.perf_counter()
                    fut = pool.submit(post_process, logits, img.shape[:2], args.min_area, args.hfov)
                    q_post.put((name, img, t0, t1, fut))
                q_post.put(stop)

            workers = [threading.Thread(target=f, daemon=True) for f in (reader, infer)]
            for t in workers:
                t.start()
            try:
                while (it := q_post.get()) is not stop:
                    name, img, t0, t1, fut = it
                    mask, obstacles = fut.result()
                    t2 = time.perf_counter()
                    times["pre+infer"].append(t1 - t0); times["post"].append(t2 - t1); times["total"].append(t2 - t0)
                    if not handle(name, img, mask, obstacles):
                        break
            finally:
                pool.shutdown(cancel_futures=True)
    finally:
        wall = time.perf_counter() - wall0
        seg.close()
        if log:
            log.close()

    if times["total"]:
        summary = {k: {"mean_ms": 1000 * float(np.mean(v[1:] or v)), "p95_ms": 1000 * float(np.percentile(v, 95))}
                   for k, v in times.items()}
        summary["mode"] = "sequential" if args.sequential else "pipelined"
        # sequential: 1/độ trễ mỗi khung; pipelined: "post" gồm cả thời gian chờ tiến trình con
        # fps_throughput = số khung / thời gian thực (gồm cả đọc và ghi file)
        summary["fps_end_to_end"] = 1000 / summary["total"]["mean_ms"]
        summary["fps_throughput"] = len(times["total"]) / wall
        summary["frames"] = len(times["total"])
        print(json.dumps(summary, indent=2))
        if out:
            (out / "timing.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
