"""Huấn luyện eWaSR / eWaSR-H trên MaSTr1325.

    python tools/train.py --config configs/ewasr_h.yaml --out runs/ewasr_h
    python tools/train.py --config configs/ewasr_h.yaml --out runs/ewasr_h --set train.epochs=50

Tự chạy tiếp từ `<out>/last.pt` nếu có (Kaggle giới hạn 12 giờ/phiên). `--time-limit-h` dừng
an toàn trước khi phiên hết giờ. Kết quả: last.pt (đầy đủ trạng thái), best.pth (chỉ trọng số,
theo mIoU val), metrics.csv, config.yaml.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wars.config import load_config  # noqa: E402
from wars.data.mastr import MaSTr1325  # noqa: E402
from wars.data.transforms import eval_transform, train_transform  # noqa: E402
from wars.losses import focal_loss, water_obstacle_separation_loss  # noqa: E402
from wars.models import get_model, load_weights  # noqa: E402

CLASSES = ("obstacle", "water", "sky")
LOG_FIELDS = ["epoch", "loss", "focal", "sep", "lr", "time_min", "val_loss",
              *[f"iou_{c}" for c in CLASSES], "miou"]


def build_optimizer(model, t):
    enc, dec_w, dec_b = [], [], []
    for name, p in model.named_parameters():
        if name.startswith("backbone"):
            enc.append(p)
        elif "weight" in name:
            dec_w.append(p)
        else:
            dec_b.append(p)
    opt = torch.optim.RMSprop([
        {"params": enc, "lr": t["lr"]},
        {"params": dec_w, "lr": t["lr"] * 10},
        {"params": dec_b, "lr": t["lr"] * 20},
    ], momentum=t["momentum"], alpha=0.9, weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (1 - e / t["epochs"]) ** t["lr_decay_pow"])
    return opt, sched


@torch.no_grad()
def validate(model, dl, device):
    model.eval()
    inter, union = np.zeros(3), np.zeros(3)
    loss_sum, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += focal_loss(logits, y).item() * len(x)
        n += len(x)
        logits = torch.nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                                 align_corners=False)
        pred, gt = logits.argmax(1), y.argmax(1)
        valid = y.sum(1) > 0.9
        for c in range(3):
            p, g = (pred == c) & valid, (gt == c) & valid
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    iou = inter / np.maximum(union, 1)
    return {"val_loss": loss_sum / max(n, 1), **{f"iou_{c}": iou[i] for i, c in enumerate(CLASSES)},
            "miou": iou.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override, vd. train.epochs=50")
    ap.add_argument("--time-limit-h", type=float, default=None, help="dừng an toàn sau N giờ")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    m, d, t = cfg["model"], cfg["data"], cfg["train"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    random.seed(t["seed"]); np.random.seed(t["seed"]); torch.manual_seed(t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(m["input_size"])
    root = Path(d["mastr_root"])

    train_ds = MaSTr1325(root / d["image_dir"], root / d["mask_dir"], d["train_list"],
                         train_transform(size, d.get("vn_color_aug", False)))
    val_ds = MaSTr1325(root / d["image_dir"], root / d["mask_dir"], d["val_list"], eval_transform(size))
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                          num_workers=t["workers"], pin_memory=True, persistent_workers=t["workers"] > 0)
    val_dl = DataLoader(val_ds, batch_size=t["batch_size"], num_workers=t["workers"])

    model = get_model(m["name"]).to(device)
    opt, sched = build_optimizer(model, t)
    scaler = torch.amp.GradScaler(enabled=t["amp"] and device.type == "cuda")

    start_epoch, best = 0, -1.0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"Tiếp tục từ epoch {start_epoch} (best mIoU {best:.4f})")
    elif m.get("init_weights"):
        skipped = load_weights(model, m["init_weights"], strict=False)
        print(f"Khởi tạo từ {m['init_weights']}; {len(skipped)} tham số học từ đầu")

    t0 = time.time()
    log_path = out / "metrics.csv"
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        tot, tot_fl, tot_sl, n = 0.0, 0.0, 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                logits, aux = model(x, return_aux=True)
            # Loss tính ở fp32 cho ổn định số học
            fl = focal_loss(logits.float(), y)
            sl = t["separation_loss_lambda"] * water_obstacle_separation_loss(aux.float(), y)
            loss = fl + sl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); tot_fl += fl.item(); tot_sl += sl.item(); n += 1
        sched.step()

        row = {"epoch": epoch, "loss": tot / n, "focal": tot_fl / n, "sep": tot_sl / n,
               "lr": opt.param_groups[0]["lr"], "time_min": (time.time() - t0) / 60}
        if (epoch + 1) % t["val_every"] == 0 or epoch + 1 == t["epochs"]:
            row.update(validate(model, val_dl, device))
            if row["miou"] > best:
                best = row["miou"]
                torch.save(model.state_dict(), out / "best.pth")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)

        new = not log_path.exists()
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="")
            if new:
                w.writeheader()
            w.writerow(row)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best}, last)

        if args.time_limit_h and (time.time() - t0) / 3600 > args.time_limit_h:
            print(f"Hết thời gian cho phép sau epoch {epoch}; chạy lại cùng lệnh để tiếp tục.")
            return

    torch.save(model.state_dict(), out / "final.pth")
    print(f"Xong. best mIoU={best:.4f}")


if __name__ == "__main__":
    main()
