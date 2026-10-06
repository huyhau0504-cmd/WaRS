# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Trả lời người dùng bằng tiếng Việt.**

## Dự án

Đồ án + bài báo: phân đoạn vật cản trên mặt nước (nước / trời / vật cản) cho USV, chạy trên Raspberry Pi 5 + **Hailo-8L (13 TOPS)**, dựa trên WaSR/eWaSR. Đề cương, mục tiêu, kế hoạch 8 tuần (07/10 – 06/12/2026): [docs/de_cuong.md](docs/de_cuong.md). Đóng góp chính: eWaSR-H (biến thể thân thiện Hailo), phân tích lượng tử hóa INT8, so sánh với YOLOv8n, đánh giá trên dữ liệu sông Việt Nam (USV-VN).

## Môi trường

| Nơi | Việc | Ghi chú |
|---|---|---|
| Laptop Ubuntu 26.04 (RTX 3050 4GB) | viết code, test, vẽ nhãn, đánh giá | `.venv/` (Python 3.14, **torch CPU**). `source .venv/bin/activate` |
| Kaggle (GPU) | huấn luyện eWaSR/eWaSR-H, YOLOv8n | [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) clone repo từ GitHub; dữ liệu là Kaggle private dataset |
| Docker "Hailo AI Software Suite" trên laptop | DFC: parse/optimize/compile, giả lập INT8 | DFC cần Python 3.10 và không hỗ trợ chính thức Ubuntu 26.04 → luôn chạy trong Docker |
| Raspberry Pi 5 | chạy HEF bằng HailoRT | `sudo apt install hailo-all`; chỉ cần numpy + opencv |

Không có test suite. Kiểm tra thay đổi bằng cách chạy script thật trên dữ liệu nhỏ (xem các lệnh dưới).

## Lệnh thường dùng

```bash
# Huấn luyện (tự chạy tiếp từ <out>/last.pt; override cấu hình bằng --set key.sub=value)
python tools/train.py --config configs/ewasr_h.yaml --out runs/ewasr_h --set train.epochs=50
# Smoke test huấn luyện trên CPU: trỏ data.mastr_root/train_list/val_list tới vài ảnh, model.input_size=[192,256], train.workers=0

# Dự đoán → mask PNG mã lớp (PyTorch hoặc ONNX), rồi đánh giá trên USV-VN
python tools/predict.py --model ewasr --weights weights/ewasr_resnet18.pth --split test --out preds/ewasr --vis
python tools/predict.py --onnx exports/ewasr_h_384x512.onnx --split test --out preds/ewasr_h_onnx
python tools/evaluate_usv.py --pred-masks preds/ewasr/masks --split test --size-edges 56 70
python tools/evaluate_usv.py --pred-boxes preds/yolov8n/boxes.json --split test --size-edges 56 70

# MODS: ghi mask màu theo sequence, chấm bằng MODS evaluation toolkit (github.com/bborja/mods_evaluation)
python tools/predict.py --model ewasr_h --weights runs/ewasr_h/best.pth --mods /path/to/mods --out preds/mods/ewasr_h

# Xuất ONNX (shape tĩnh, tự so sai số với PyTorch)
python tools/export_onnx.py --model ewasr_h --weights runs/ewasr_h/best.pth --size 384 512

# Hailo (trong Docker DFC)
python hailo/compile.py --onnx exports/ewasr_h_384x512.onnx --parse-only --out hailo/build/ewasr_h_parse
python hailo/compile.py --onnx exports/ewasr_h_384x512.onnx --calib-dir <mastr>/images --opt-level 2 --out hailo/build/ewasr_h_o2
python hailo/infer_emulator.py --har hailo/build/ewasr_h_o2_quantized.har --ctx quantized --split test --out preds/ewasr_h_int8

# Raspberry Pi
python3 pi/run.py --hef ewasr_h_384x512.hef --split test --out results/ewasr_h_pi --hfov <độ>
hailortcli benchmark ewasr_h_384x512.hef   # FPS thuần của HEF

# Vẽ mép nước cho tập test (cần cho FP/Precision/F1)
python tools/annotate_water_edge.py --split test
```

Trọng số eWaSR chính thức (không IMU): `curl -L -o weights/ewasr_resnet18.pth https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth`.

## Kiến trúc

**Luồng:** `tools/train.py` (Kaggle) → `best.pth` → `tools/export_onnx.py` → `hailo/compile.py` (HAR → HAR INT8 → HEF) → `pi/run.py`. Mọi backend (PyTorch, ONNX, Hailo emulator, Pi) ghi cùng một định dạng — `<out>/masks/<stem>.png` mã lớp ở độ phân giải ảnh gốc — nên `tools/evaluate_usv.py` chấm được tất cả, và so FP32 với INT8 là so hai thư mục mask.

**Hợp đồng vào/ra của mô hình** (giữ nhất quán khi sửa):
- Lớp: 0 vật cản, 1 nước, 2 trời. Mask MaSTr1325 giá trị 4 = không xác định → one-hot toàn 0 → bị bỏ qua trong loss/metric; viền tô 0 sau Affine augmentation cũng bị bỏ qua theo cách này.
- PyTorch/ONNX nhận ảnh **đã chuẩn hóa ImageNet** NCHW. HEF nhận **RGB uint8 NHWC**, vì chuẩn hóa nằm trong model script của Hailo (`normalization1 = ...` trong `hailo/compile.py`). Hằng số ở `wars/constants.py`.
- Đầu ra là logits ở **H/4 × W/4**. Upsample nằm ngoài mô hình (`wars/postprocess.logits_to_mask`: upsample logits rồi mới argmax) để giảm tải cho Hailo.

**Mô hình (`wars/models/`)** là bản port eWaSR không IMU. **Tên thuộc tính giữ y hệt bản gốc** (`backbone`, `decoder.sim1..4`, `metaformers`, `metaformers_skip2`, `seg_head`, `convs_project*`...) để nạp được trọng số chính thức — đã kiểm tra khớp ONNX chính thức (sai số 1.7e-5). Đừng đổi tên chúng.
- Biến thể đăng ký trong `MODELS` ([wars/models/__init__.py](wars/models/__init__.py)). Chuỗi mixer: `C` = ARM (attention kênh), `S` = spatial mean/max theo trục kênh (gốc; sinh ReduceMax/ReduceMean), `D` = depthwise spatial gate (eWaSR-H, không reduce theo trục kênh). `ewasr_h_p` bật `project` → 23,9M tham số so với 60,3M — dự phòng nếu Hailo-8L không chứa nổi decoder đầy đủ.
- `init_weights` nạp `strict=False` (chỉ khóa trùng tên & shape) để khởi tạo eWaSR-H từ eWaSR chính thức.
- `resize_like`/`_pool_to` ép shape về `int` và dùng `scale_factor` nguyên / AvgPool kernel cố định để ONNX có shape tĩnh, dạng mà DFC dễ nhận. H, W đầu vào phải chia hết cho 32.

**Huấn luyện:** siêu tham số gốc của WaSR/eWaSR (RMSprop lr 1e-6, decoder ×10, bias ×20, poly decay; focal loss + water-obstacle separation loss λ=0.01 trên đặc trưng layer3). Cấu hình YAML kế thừa qua `base:` ([configs/base.yaml](configs/base.yaml)). Kaggle không giữ `/kaggle/working` giữa các phiên: dùng `--time-limit-h`, Save Version, rồi Add Input output cũ để chạy tiếp từ `last.pt`.

**Module không phụ thuộc torch:** `wars/constants.py`, `wars/postprocess.py`, `wars/data/usv_vn.py`, `wars/data/mods.py`, `wars/eval/` — được `pi/` và `hailo/` import (Pi và Docker DFC không chắc có torch/albumentations). Không thêm import torch vào các file này.

**Mã Hailo/HailoRT** (`hailo/*.py`, `pi/run.py`) viết theo API DFC 3.x / HailoRT 4.x nhưng **chưa được chạy thật** — lần đầu dùng cần đối chiếu với phiên bản đã cài.

## Dữ liệu USV-VN và giao thức đánh giá

- `data/usv_vn/{train,valid,test}/{images,labels}`: 751 khung `left_XXXX.jpg` 640×640 (đã bị resize từ ảnh gốc, không còn bản gốc), nhãn YOLO 1 lớp, đúng 1 box/ảnh (chai/can nhựa nổi). Ảnh **không** commit lên git; chỉ `data/usv_vn/water_edges.json` được commit.
- Đã **chia theo thời gian** (train 1–754, valid 863–1047, test 1059–1259) bằng `tools/resplit_by_time.py` vì các khung liên tiếp gần như giống hệt nhau. Cách chia ngẫu nhiên gốc lưu ở `splits/original_split.csv` (`--restore`). Không chia ngẫu nhiên lại.
- Tiêu chí chính là **coverage** (kiểu MODS), dùng chung cho phân đoạn và YOLO (box YOLO được tô thành mask). Tiêu chí IoU chỉ là phụ: camera đặt sát mặt nước nên vật cản đè lên dải bờ trong ảnh, mask phân đoạn dính liền với bờ, quy về box thì IoU thấp dù phát hiện đúng.
- FP/Precision/F1 chỉ tính được cho ảnh đã có mép nước trong `water_edges.json` (vẽ phía sau vật cản động, như MODS); thiếu thì chỉ có Recall. Recall đơn lẻ dễ gây hiểu nhầm: eWaSR chính thức zero-shot đạt recall 141/141 trên test nhưng báo động sai nhiều ở bóng phản chiếu của bờ.
- Theo ngưỡng COCO, mọi box valid/test đều rơi vào "medium" → dùng `--size-edges 56 70` khi phân tích theo kích thước. Ngưỡng `--coverage-thr 0.5` cần đối chiếu với MODS toolkit trước khi đưa số vào bài báo.

## Nguồn gốc mã

Port từ eWaSR/WaSR (Apache-2.0) — xem [NOTICE](NOTICE). Giữ header ghi nguồn ở các file đã port.
