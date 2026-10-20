"""Chạy các thí nghiệm trong configs/kaggle_run.yaml. Notebook Kaggle chỉ clone repo rồi gọi script này.

    python tools/kaggle_run.py                      # mặc định: /kaggle/input → /kaggle/working
    python tools/kaggle_run.py --input-root /tmp/in --work /tmp/work --run-config configs/kaggle_run.yaml

Mỗi run ghi vào <work>/runs/<name>/: last.pt, best.pth, metrics.csv, train.log, run_info.json và,
nếu bật, <name>_<H>x<W>.onnx, pred_usv_test/, usv_test.json. Tổng hợp ở <work>/STATUS.json; dòng
cuối là `KAGGLE_RUN_STATUS: ok|partial|failed`. Mã thoát khác 0 chỉ khi không run nào có
checkpoint, để notebook không lỗi sau khi đã có kết quả đáng giữ.
"""
import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wars.config import load_config  # noqa: E402

SPLITS = ("train", "valid", "test")
_print_lock = threading.Lock()
_pip_lock = threading.Lock()


def say(msg):
    with _print_lock:
        print(msg, flush=True)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def find_data(input_root):
    """Tìm MaSTr1325 (theo `*m.png`) và USV-VN (theo `<split>/images/left_*.jpg`) dưới input_root."""
    root = Path(input_root)
    mastr = None
    for mask in root.rglob("*m.png"):
        if mask.parents[1].name in SPLITS:
            continue  # mask USV-VN (<split>/masks/left_XXXXm.png, tools/make_usv_masks.py), không phải MaSTr1325
        img = next(root.rglob(mask.name[:-5] + ".jpg"), None)
        if img is None:
            continue
        mroot = Path(os.path.commonpath([mask.parent, img.parent]))
        mastr = {"mastr_root": str(mroot), "image_dir": str(img.parent.relative_to(mroot)),
                 "mask_dir": str(mask.parent.relative_to(mroot))}
        # Danh sách ảnh đi kèm dữ liệu (vd. dữ liệu giả) thay cho configs/mastr1325/*.txt
        for key, fname in (("train_list", "train.txt"), ("val_list", "val.txt")):
            if (mroot / fname).exists():
                mastr[key] = str(mroot / fname)
        break
    usv = next((str(p.parents[2]) for p in root.rglob("left_*.jpg")
                if p.parent.name == "images" and p.parents[1].name in SPLITS), None)
    return mastr, usv


def find_previous(input_root, name):
    """Thư mục runs/<name> có last.pt trong output của version trước (đã Add Input)."""
    return next((p.parent for p in Path(input_root).rglob(f"runs/{name}/last.pt")), None)


def stream(cmd, log_path, name, env):
    """Chạy lệnh trong ROOT; ghi stdout+stderr vào log_path và in ra với tiền tố [name]."""
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(map(str, cmd))}\n")
        p = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            log.flush()
            say(f"[{name}] {line.rstrip()}")
        return p.wait()


def metrics(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def gpu_name(slot):
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(slot)],
                              capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class Runner:
    def __init__(self, args, rc):
        import torch
        self.args, self.rc = args, rc
        self.work = Path(args.work)
        self.t0 = time.time()
        self.n_gpu = torch.cuda.device_count()
        self.n_par = max(1, self.n_gpu)
        self.workers = max(1, (os.cpu_count() or 1) // self.n_par)
        self.versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                         "cuda": torch.version.cuda}
        self.commit, self.branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
        self.dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
        self.mastr, self.usv = find_data(args.input_root)

    def remaining_h(self):
        return max(0.05, self.rc.get("time_limit_h", 11) - (time.time() - self.t0) / 3600)

    def env(self, slot):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if self.n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        return env

    def run(self, spec, slot):
        name = spec["name"]
        out = self.work / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        info = {"name": name, "kind": spec.get("kind", "seg"), "config": spec.get("config"),
                "set": list(spec.get("set", [])), "status": "running", "has_checkpoint": False,
                "commit": self.commit, "branch": self.branch, "dirty": self.dirty, **self.versions,
                "gpu": gpu_name(slot) if self.n_gpu else None, "started": now(), "post": {}}
        self.write_info(out, info)
        try:
            prev = find_previous(self.args.input_root, name)
            if prev and not (out / "last.pt").exists():
                shutil.copytree(prev, out, dirs_exist_ok=True)
                say(f"[{name}] chép checkpoint từ {prev} để chạy tiếp")
            if info["kind"] == "seg":
                self.run_seg(spec, slot, out, info)
            elif info["kind"] == "yolo":
                self.run_yolo(spec, slot, out, info)
            else:
                raise ValueError(f"kind không hợp lệ: {info['kind']}")
        except Exception as e:  # một run lỗi không được làm dừng các run khác
            info["status"], info["error"] = "failed", f"{type(e).__name__}: {e}"
            say(f"[{name}] LỖI: {info['error']}")
        info["finished"] = now()
        self.write_info(out, info)
        return info

    @staticmethod
    def write_info(out, info):
        (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    def post_step(self, info, step, cmd, out, name, env):
        rc = stream(cmd, out / "train.log", name, env)
        info["post"][step] = "ok" if rc == 0 else f"lỗi (mã {rc})"

    def run_seg(self, spec, slot, out, info):
        name = spec["name"]
        if self.mastr is None:
            raise FileNotFoundError(f"không tìm thấy MaSTr1325 (*m.png) dưới {self.args.input_root}")
        sets = ([f"data.{k}={v}" for k, v in self.mastr.items()] + [f"train.workers={self.workers}"]
                + list(spec.get("set", [])))
        cfg = load_config(ROOT / spec["config"], sets)
        info["merged_config"] = cfg
        n_before = len(metrics(out / "metrics.csv"))
        t0 = time.time()
        rc = stream([sys.executable, "tools/train.py", "--config", spec["config"], "--out", out,
                     "--time-limit-h", f"{self.remaining_h():.3f}", "--set", *sets],
                    out / "train.log", name, self.env(slot))
        rows = metrics(out / "metrics.csv")
        n_new = len(rows) - n_before
        info.update(returncode=rc, epochs_total=len(rows), epochs_this_session=n_new,
                    sec_per_epoch=round((time.time() - t0) / n_new, 1) if n_new > 0 else None,
                    best_miou=max((float(r["miou"]) for r in rows if r.get("miou")), default=None),
                    has_checkpoint=(out / "last.pt").exists())
        info["status"] = ("failed" if rc != 0 else "done" if (out / "final.pth").exists() else "time_limit")
        if not (out / "best.pth").exists():
            return

        model, (h, w) = cfg["model"]["name"], cfg["model"]["input_size"]
        env = self.env(slot)
        if spec.get("export_onnx", self.rc.get("export_onnx", True)):
            self.post_step(info, "export_onnx", [sys.executable, "tools/export_onnx.py", "--model", model,
                                                 "--weights", out / "best.pth", "--size", h, w,
                                                 "--out", out / f"{name}_{h}x{w}.onnx"], out, name, env)
        if spec.get("eval_usv", self.rc.get("eval_usv", True)) and self.usv:
            pred = out / "pred_usv_test"
            self.post_step(info, "predict_usv", [sys.executable, "tools/predict.py", "--model", model,
                                                 "--weights", out / "best.pth", "--size", h, w,
                                                 "--usv-root", self.usv, "--split", "test", "--out", pred],
                           out, name, env)
            self.post_step(info, "evaluate_usv", [sys.executable, "tools/evaluate_usv.py", "--pred-masks",
                                                  pred / "masks", "--usv-root", self.usv, "--split", "test",
                                                  "--size-edges", 56, 70, "--out-json", out / "usv_test.json"],
                           out, name, env)

    def run_yolo(self, spec, slot, out, info):
        name = spec["name"]
        if self.usv is None:
            raise FileNotFoundError(f"run yolo cần USV-VN (left_*.jpg) dưới {self.args.input_root}")
        with _pip_lock:
            if shutil.which("yolo") is None:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
        data = yaml.safe_load((ROOT / "configs/usv_vn_yolo.yaml").read_text())
        data["path"] = self.usv
        data_yaml = out / "usv_vn_yolo.yaml"
        data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True))
        env = self.env(slot)
        args = ["model=yolov8n.pt", "imgsz=640", "epochs=100", f"time={self.remaining_h():.3f}",
                f"device={0 if self.n_gpu else 'cpu'}", *spec.get("set", [])]
        rc = stream(["yolo", "detect", "train", f"data={data_yaml}", f"project={out.parent}", f"name={name}",
                     "exist_ok=True", *args], out / "train.log", name, env)
        best = out / "weights" / "best.pt"
        info.update(returncode=rc, has_checkpoint=best.exists() or (out / "weights" / "last.pt").exists(),
                    status="done" if rc == 0 and best.exists() else "failed")
        if not best.exists():
            return
        pred = out / "pred_usv_test"
        self.post_step(info, "predict_usv", [sys.executable, "tools/yolo_predict.py", "--weights", best,
                                             "--usv-root", self.usv, "--split", "test", "--out", pred],
                       out, name, env)
        self.post_step(info, "evaluate_usv", [sys.executable, "tools/evaluate_usv.py", "--pred-boxes",
                                              pred / "boxes.json", "--usv-root", self.usv, "--split", "test",
                                              "--size-edges", 56, 70, "--out-json", out / "usv_test.json"],
                       out, name, env)
        if spec.get("export_onnx", self.rc.get("export_onnx", True)):
            # ONNX cho Hailo Model Zoo (hailomz compile yolov8n --ckpt ... --hw-arch hailo8l --classes 1)
            self.post_step(info, "export_onnx", ["yolo", "export", f"model={best}", "format=onnx", "opset=11",
                                                 "imgsz=640"], out, name, env)

    def main(self):
        runs = self.rc.get("runs", [])
        say(f"commit {self.commit} ({self.branch}) · GPU: {self.n_gpu} · song song: {self.n_par} · "
            f"workers/run: {self.workers}")
        say(f"MaSTr1325: {self.mastr}")
        say(f"USV-VN: {self.usv}")
        if self.usv is None:
            say("CẢNH BÁO: không tìm thấy USV-VN → bỏ qua đánh giá USV-VN; run yolo sẽ lỗi")
        if self.mastr is None and any(r.get("kind", "seg") == "seg" for r in runs):
            say(f"LỖI: không tìm thấy MaSTr1325 dưới {self.args.input_root} → các run seg bị bỏ qua")

        slots = queue.Queue()
        for i in range(self.n_par):
            slots.put(i)

        def task(spec):
            slot = slots.get()
            try:
                return self.run(spec, slot)
            finally:
                slots.put(slot)

        with ThreadPoolExecutor(max_workers=self.n_par) as pool:
            results = list(pool.map(task, runs))

        status = ("failed" if not any(r["has_checkpoint"] for r in results) else
                  "ok" if all(r["status"] == "done" for r in results) else "partial")
        summary = {"status": status, "commit": self.commit, "branch": self.branch, "dirty": self.dirty,
                   "elapsed_h": round((time.time() - self.t0) / 3600, 3), "finished": now(),
                   "n_gpu": self.n_gpu, "data": {"mastr": self.mastr, "usv": self.usv},
                   "runs": [{k: r.get(k) for k in ("name", "kind", "status", "has_checkpoint", "epochs_total",
                                                   "sec_per_epoch", "best_miou", "post", "error")}
                            for r in results]}
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "STATUS.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        for r in summary["runs"]:
            say(f"  {r['name']}: {r['status']} · epoch {r['epochs_total']} · best mIoU {r['best_miou']}"
                + (f" · {r['error']}" if r.get("error") else ""))
        say(f"KAGGLE_RUN_STATUS: {status}")
        return 1 if status == "failed" else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--run-config", default="configs/kaggle_run.yaml")
    args = ap.parse_args()
    rc = yaml.safe_load((ROOT / args.run_config).read_text()) or {}
    sys.exit(Runner(args, rc).main())


if __name__ == "__main__":
    main()
