# WaRS — Nhận diện vật cản trên mặt nước cho USV trên Raspberry Pi 5 + Hailo-8L

Phân đoạn ảnh thành **nước / trời / vật cản** theo thời gian thực, dựa trên [eWaSR](https://github.com/tersekmatija/eWaSR), tối ưu cho bộ tăng tốc Hailo-8L. Đề cương đầy đủ: [docs/de_cuong.md](docs/de_cuong.md).

## Cấu trúc

```
wars/            thư viện: mô hình (eWaSR, eWaSR-H), loss, dữ liệu, hậu xử lý, đánh giá
tools/           train, predict, export_onnx, evaluate_usv, yolo_predict, annotate_water_edge, resplit_by_time
hailo/           biên dịch ONNX → HEF (Hailo DFC) và giả lập INT8
pi/              pipeline chạy trên Raspberry Pi 5 + HailoRT
configs/         cấu hình huấn luyện (YAML), danh sách ảnh MaSTr1325, dataset YOLO
notebooks/       notebook huấn luyện trên Kaggle
data/usv_vn/     bộ dữ liệu USV-VN (ảnh không đưa lên git)
```

## Bắt đầu

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # laptop: CPU là đủ
pip install -r requirements.txt
mkdir -p weights && curl -L -o weights/ewasr_resnet18.pth \
  https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth

# Thử eWaSR chính thức trên tập test USV-VN
python tools/predict.py --model ewasr --weights weights/ewasr_resnet18.pth --split test --out preds/ewasr --vis
python tools/evaluate_usv.py --pred-masks preds/ewasr/masks --split test --size-edges 56 70
```

Huấn luyện trên Kaggle: mở [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb). Các lệnh khác xem [CLAUDE.md](CLAUDE.md).

## Dữ liệu

- **MaSTr1325** (huấn luyện) và **MODS** (đánh giá): tải từ trang của nhóm ViCoS, Đại học Ljubljana.
- **USV-VN**: 751 ảnh sông đô thị Việt Nam, nhãn bounding box, chia train/valid/test theo thời gian.

## Giấy phép mã nguồn gốc

Phần mô hình và loss được chuyển thể từ eWaSR và WaSR (Apache-2.0), xem [NOTICE](NOTICE).
