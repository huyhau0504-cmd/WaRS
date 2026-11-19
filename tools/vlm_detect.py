"""Phát hiện vật cản trên MODS bằng mô hình thị giác–ngôn ngữ (so sánh với eWaSR-Lite theo đề nghị của thầy).

    python tools/vlm_detect.py --model owlv2 --mods-root /path/mods --sequences-file configs/mods_vlm_subset.txt --out out/owlv2
    python tools/vlm_detect.py --model paligemma2 --model-path /kaggle/input/paligemma-2/... ...
    python tools/vlm_detect.py --model qwen25vl ...
    python tools/vlm_detect.py ... --limit 5          # thử nhanh

Mô hình (mặc định tải từ Hugging Face; --model-path để dùng bản đã có, ví dụ Kaggle Models):
  owlv2       google/owlv2-base-patch16-ensemble — ViT của Google + bộ mã hóa văn bản, phát hiện theo danh sách tên
  paligemma2  google/paligemma2-3b-mix-448      — ViT SigLIP (Google) + mô hình ngôn ngữ Gemma 2B, lệnh "detect ..."
  qwen25vl    Qwen/Qwen2.5-VL-3B-Instruct       — ViT + mô hình ngôn ngữ Qwen2.5 3B, yêu cầu trả khung bao JSON

Đầu ra <out>/boxes.json: {"meta": {...}, "frames": {"<seq>/<frame>": [[x1, y1, x2, y2, score, label], ...]}} theo pixel
ảnh gốc (1278×958); <out>/raw.jsonl: câu trả lời thô của mô hình (VLM) để kiểm tra; thời gian xử lý từng ảnh
(không tính nạp mô hình). Chạy tiếp được: khung đã có trong boxes.json được bỏ qua. Chấm điểm: tools/boxes_to_mods.py.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

NAMES = ["boat", "ship", "sailboat", "kayak", "buoy", "person", "swimmer", "floating object"]
DEFAULT_IDS = {"owlv2": "google/owlv2-base-patch16-ensemble", "paligemma2": "google/paligemma2-3b-mix-448",
               "qwen25vl": "Qwen/Qwen2.5-VL-3B-Instruct"}
QWEN_PROMPT = ("Detect every obstacle on or in the water that a small boat must avoid: boats, ships, sailboats, "
               "kayaks, buoys, people, swimmers and any other floating object. Ignore the shore, piers, buildings "
               "and the sky. Output a JSON list where each item is {\"bbox_2d\": [x1, y1, x2, y2], \"label\": name}.")


def mods_frames(root, seq_ids):
    """(khóa "<seq>/<frame>", đường dẫn ảnh) cho các sequence (chỉ số từ 1 theo thứ tự trong mods.json)."""
    seqs = json.loads((root / "mods.json").read_text())["dataset"]["sequences"]
    out = []
    for i in seq_ids:
        name = seqs[i - 1]["path"].strip("/").split("/")[0]
        for f in seqs[i - 1]["frames"]:
            out.append((f"{name}/{Path(f['image_file_name']).stem}", root / "sequences" / name / "frames" / f["image_file_name"]))
    return out


class OWLv2:
    def __init__(self, path, device, thr):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(path)
        self.model = Owlv2ForObjectDetection.from_pretrained(path).to(device).eval()
        self.device, self.thr = device, thr

    @torch.no_grad()
    def __call__(self, img):
        inp = self.proc(text=[[f"a photo of a {n}" for n in NAMES]], images=img, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        # OWLv2 đệm ảnh thành hình vuông ở góc trên trái → hộp tính theo cạnh lớn nhất
        s = max(img.size)
        sizes = torch.tensor([[s, s]], device=self.device)
        if hasattr(self.proc, "post_process_grounded_object_detection"):  # transformers ≥ 4.50
            r = self.proc.post_process_grounded_object_detection(out, threshold=self.thr, target_sizes=sizes)[0]
        else:
            r = self.proc.post_process_object_detection(out, threshold=self.thr, target_sizes=sizes)[0]
        boxes = [[*map(float, b), float(sc), NAMES[int(l)]] for b, sc, l in zip(r["boxes"], r["scores"], r["labels"])]
        return boxes, None


class PaliGemma2:
    LOC = re.compile(r"((?:<loc\d{4}>){4})\s*([^;<]*)")

    def __init__(self, path, device, thr):
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        self.proc = PaliGemmaProcessor.from_pretrained(path)
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        self.device, self.dtype = device, dtype
        self.prompt = "<image>detect " + " ; ".join(NAMES)

    @torch.no_grad()
    def __call__(self, img):
        inp = self.proc(text=self.prompt, images=img, return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
        n = inp["input_ids"].shape[-1]
        gen = self.model.generate(**inp, max_new_tokens=256, do_sample=False)
        text = self.proc.decode(gen[0][n:], skip_special_tokens=False).replace("<eos>", "")
        W, H = img.size
        boxes = []
        for locs, label in self.LOC.findall(text):
            y1, x1, y2, x2 = [int(v) / 1024 for v in re.findall(r"\d{4}", locs)]  # ảnh được co về 448×448
            boxes.append([x1 * W, y1 * H, x2 * W, y2 * H, 1.0, label.strip()])
        return boxes, text


class Qwen25VL:
    def __init__(self, path, device, thr):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(path)
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if device == "cuda" else torch.float32)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QWEN_PROMPT}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[img], return_tensors="pt").to(self.device)
        n = inp["input_ids"].shape[-1]
        gen = self.model.generate(**inp, max_new_tokens=1024, do_sample=False)
        ans = self.proc.batch_decode(gen[:, n:], skip_special_tokens=True)[0]
        # Qwen2.5-VL trả tọa độ tuyệt đối theo ảnh đã co về bội số của 28 → quy về ảnh gốc
        _, gh, gw = inp["image_grid_thw"][0].tolist()
        sx, sy = img.size[0] / (gw * 14), img.size[1] / (gh * 14)
        boxes = []
        for item in parse_json_list(ans):
            b = item.get("bbox_2d") if isinstance(item, dict) else None
            if isinstance(b, (list, tuple)) and len(b) == 4:
                x1, y1, x2, y2 = map(float, b)
                boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy, 1.0, str(item.get("label", ""))])
        return boxes, ans


def parse_json_list(s):
    """Danh sách JSON đầu tiên trong câu trả lời (có thể nằm trong ```json ... ```); [] nếu không đọc được."""
    m = re.search(r"\[.*\]", s, re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return [json.loads(x) for x in re.findall(r"\{[^{}]*\}", m.group(0)) if _ok(x)]


def _ok(x):
    try:
        json.loads(x)
        return True
    except json.JSONDecodeError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DEFAULT_IDS))
    ap.add_argument("--model-path", help="thư mục hoặc id Hugging Face (mặc định: id chuẩn của từng mô hình)")
    ap.add_argument("--mods-root", required=True)
    ap.add_argument("--sequences-file", default="configs/mods_vlm_subset.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.05, help="ngưỡng điểm tối thiểu của OWLv2 (lọc lại khi chấm)")
    ap.add_argument("--limit", type=int, help="chỉ chạy N khung đầu (thử nhanh)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root, out = Path(args.mods_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq_ids = [int(x) for x in Path(args.sequences_file).read_text().split()]
    frames = mods_frames(root, seq_ids)[:args.limit]
    res_path = out / "boxes.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {"meta": {}, "frames": {}}

    t0 = time.time()
    model = {"owlv2": OWLv2, "paligemma2": PaliGemma2, "qwen25vl": Qwen25VL}[args.model](
        args.model_path or DEFAULT_IDS[args.model], device, args.thr)
    load_s = time.time() - t0
    times = res["meta"].get("seconds_per_image", [])
    with open(out / "raw.jsonl", "a") as raw:
        for k, (key, path) in enumerate(frames):
            if key in res["frames"]:
                continue
            img = Image.open(path).convert("RGB")
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            boxes, text = model(img)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(round(time.time() - t, 4))
            res["frames"][key] = boxes
            if text is not None:
                raw.write(json.dumps({"frame": key, "text": text}, ensure_ascii=False) + "\n")
            if k % 50 == 0 or k == len(frames) - 1:
                res["meta"] = {"model": args.model, "model_path": args.model_path or DEFAULT_IDS[args.model],
                               "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                               "torch": torch.__version__, "load_seconds": round(load_s, 1),
                               "seconds_per_image": times, "frames_done": len(res["frames"]), "frames_total": len(frames)}
                res_path.write_text(json.dumps(res))
                med = sorted(times)[len(times) // 2] if times else 0
                print(f"{len(res['frames'])}/{len(frames)} khung · trung vị {med:.2f} s/ảnh · {len(boxes)} hộp", flush=True)
    print(f"Xong → {res_path}")


if __name__ == "__main__":
    main()
