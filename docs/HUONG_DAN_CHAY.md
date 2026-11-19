# Hướng dẫn tự chạy (laptop · Kaggle · Raspberry Pi)

Quy trình đầy đủ: **huấn luyện (Kaggle) → xuất ONNX (laptop) → lượng tử + compile HEF (laptop, Hailo DFC) → chạy trên Pi → chấm MODS (laptop)**.
Mọi lệnh chạy từ thư mục repo `~/WaRS` trên laptop, trừ khi ghi khác. Lệnh nào chạy lâu (> 10 phút) thì bọc bằng
`systemd-inhibit --what=idle:sleep:handle-lid-switch <lệnh>` để máy không ngủ, và **không tắt máy** khi đang chạy.

## 0. Môi trường (một lần)

| Ở đâu | Cái gì | Kiểm tra |
|---|---|---|
| Laptop | `.venv/` (Python 3.14, torch CPU) cho code chính | `source .venv/bin/activate && python tools/smoke_test.py` → `PASS` (~1 phút) |
| Laptop | `~/hailo-dfc/.venv` (Python 3.10, Hailo DFC 3.34) cho lượng tử/compile | `~/hailo-dfc/.venv/bin/python -c "import hailo_sdk_client"` |
| Laptop | `third_party/.venv-mods` (toolkit MODS, matplotlib) | tự tạo khi chạy `tools/eval_mods.py` lần đầu |
| Laptop | `third_party/.venv-docx` (python-docx) | chỉ để dựng file Word bài báo |
| Dữ liệu | `~/datasets/mods` (MODS), `~/datasets/lars` (LaRS), `data/mastr1325` (MaSTr1325) | đã tải sẵn |
| Pi 5 | `ssh an@raspberrypi.local`, repo ở `~/WaRS` với `.venv`, HEF ở `~/WaRS/hef/`, MODS ở `~/datasets/mods` | `hailortcli fw-control identify` |
| Kaggle | notebook `hauhuytran/wars-train` (không sửa ô nào); dataset private `mastr1325`, `lasrdata`, `mods-vlm-subset` | — |

Trọng số eWaSR công bố (không IMU), nếu cần: `curl -L -o weights/ewasr_resnet18.pth https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth`.

Mã mô hình: `ewasr` (E1, eWaSR), `ewasr_h` (E2, D mixer), `ewasr_lite` (E6, Lite S mixer), **`ewasr_h_lite` (E7, eWaSR-Lite)**.

## 1. Huấn luyện trên Kaggle

1. Sửa `configs/kaggle_run.yaml` — danh sách `runs`, mỗi run một dòng:
   ```yaml
   runs:
     - name: e7_ewasr_h_lite_s3          # tên thư mục kết quả
       kind: seg
       config: configs/ewasr_h.yaml
       set: ["model.name=ewasr_h_lite", "model.init_weights=null", "train.seed=3"]
   ```
   Tùy chọn hay dùng trong `set`: `train.epochs=100`, `train.seed=N`, `data.dataset=lars` (huấn luyện trên LaRS),
   công thức gốc eWaSR: `train.amp=false`, `train.batch_size=16`, `train.epochs=200`, `train.early_stop_patience=20`.
   Run so sánh VLM: `kind: vlm`, `model: owlv2 | qwen25vl | paligemma2` (xem `configs/kaggle_run.yaml`).
2. `python tools/smoke_test.py` → phải `PASS`.
3. Commit trên nhánh `task/...`, merge vào `main`, `git push` (notebook clone `main`).
4. Kaggle → notebook `wars-train` → kiểm tra **Input** (MaSTr1325; LaRS nếu `data.dataset=lars`; `mods-vlm-subset` + model
   PaliGemma 2 nếu chạy VLM) → **Save Version → Save & Run All (Commit) → Save**. Accelerator **GPU T4 x2**, Internet **On**.
   Hai run chạy song song (mỗi GPU một run).
5. Xong (trang notebook hiện "Version N of N"): tab **Output → Download** → giải nén vào `~/Downloads/results_XX`.
   Mỗi run có `runs/<name>/{best.pth, final.pth, <name>_384x512.onnx, metrics.csv, train.log, run_info.json}`.
6. Hết 11 giờ mà chưa xong: Save Version lần nữa, **Add Input** output của version trước → run tự chạy tiếp từ `last.pt`.

Chạy thử huấn luyện trên laptop (CPU, chậm, chỉ để kiểm tra):
```bash
python tools/train.py --config configs/ewasr_h.yaml --out runs/thu --set model.name=ewasr_h_lite train.epochs=2 train.workers=0
```

## 2. Dự đoán và chấm điểm trên laptop

```bash
source .venv/bin/activate
# ONNX (nếu Kaggle chưa xuất, hoặc xuất từ checkpoint khác)
python tools/export_onnx.py --model ewasr_h_lite --weights ~/Downloads/results_XX/runs/<run>/best.pth --size 384 512 \
    --out exports/<run>_384x512.onnx

# MODS: dự đoán (≈16 phút với ONNX) rồi chấm bằng toolkit chính thức (≈25 phút)
python tools/predict.py --onnx exports/<run>_384x512.onnx --mods ~/datasets/mods --out preds/mods/<run>_fp32
python tools/eval_mods.py --pred preds/mods/<run>_fp32 --mods-root ~/datasets/mods --out-json results/mods/<run>_fp32.json --stats
python3 tools/mods_table.py          # gộp mọi kết quả → docs/data/mods_results.csv

# LaRS val (sông/hồ và biển)
python tools/predict.py --onnx exports/<run>_384x512.onnx --images ~/datasets/lars/val/images --out preds/lars/<run>
python tools/eval_lars.py --pred-masks preds/lars/<run>/masks --scene river_like --out-json results/lars/<run>_river_like.json
python tools/eval_lars.py --pred-masks preds/lars/<run>/masks --scene sea_like   --out-json results/lars/<run>_sea_like.json
```
Kết quả MODS: `F1` (mọi vật cản), `vùng nguy hiểm F1` (vật cản trong 15 m — chỉ số chính), `mép nước RMSE`.

## 3. Lượng tử hóa và compile cho Hailo-8L (laptop)

Giao thức chính thức: **64 ảnh calib, tắt equalization, compiler max**. Hai bước tách riêng; compile `max` mất 1–4 giờ,
**không chạy hai việc DFC cùng lúc** (hết RAM).
```bash
PY=~/hailo-dfc/.venv/bin/python
# bước 1: lượng tử INT8 (≈5–15 phút) → hailo/build/<ten>_quantized.har
$PY hailo/compile.py --onnx exports/<run>_384x512.onnx --calib-dir data/mastr1325/images --calib-n 64 --opt-level 0 \
    --alls-extra hailo/no_bn_checker.alls --no-equalization --no-compile --out hailo/build/<ten>
#   (--calib-seed 1 / 2 để thử bộ ảnh calib khác)
# bước 2: compile (1–4 giờ) → hailo/build/<ten>max.hef
$PY hailo/compile.py --onnx exports/<run>_384x512.onnx --from-quantized --quantized-har hailo/build/<ten>_quantized.har \
    --alls-extra hailo/perf_max.alls --out hailo/build/<ten>max
# xem trước INT8 trên laptop (giả lập, trùng Pi ~99,6%)
$PY hailo/infer_emulator.py --har hailo/build/<ten>_quantized.har --ctx quantized --mods-sequences 35 --out preds/emu/<ten>
```

## 4. Chạy trên Raspberry Pi 5 + Hailo-8L

```bash
scp hailo/build/<ten>max.hef an@raspberrypi.local:WaRS/hef/          # từ laptop
ssh an@raspberrypi.local
cd ~/WaRS
hailortcli parse-hef hef/<ten>max.hef        # số context
hailortcli benchmark -t 15 hef/<ten>max.hef  # FPS và độ trễ trên chip
# FPS đầu-cuối (1 lượt làm nóng + 3 lượt đo, trung vị) trên 303 khung MODS
.venv/bin/python pi/run.py --hef hef/<ten>max.hef --images ~/datasets/mods_timing --out results/e2e_mods/<ten>
# MODS INT8 (≈20 phút) — mask màu MODS, rồi chép về laptop để chấm
.venv/bin/python pi/run.py --hef hef/<ten>max.hef --mods ~/datasets/mods --out results/mods/<ten>_int8
# video / camera (hướng vật cản cần góc nhìn ngang camera --hfov)
.venv/bin/python pi/run.py --hef hef/<ten>max.hef --video <file.mp4|0> --out results/video_<ten> --hfov 70 --vis
```
Trên laptop: `rsync -a --exclude timing.json an@raspberrypi.local:WaRS/results/mods/<ten>_int8 preds/mods/`, rồi chấm như
mục 2 (`tools/eval_mods.py --pred preds/mods/<ten>_int8 ...`). Tắt Pi: `sudo shutdown -h now`.

## 5. So sánh với mô hình thị giác–ngôn ngữ (VLM)

Tập con MODS: `configs/mods_vlm_subset.txt` (12 sequence, 1161 khung; dataset Kaggle `mods-vlm-subset`).
```bash
# sau khi tải output Kaggle (runs/vlm_*/boxes.json) về ~/Downloads/results_VLM
D=~/Downloads/results_VLM/runs MODELS="vlm_paligemma2 vlm_qwen25vl vlm_owlv2" THR=0.2 tools/score_vlm.sh   # → results/mods_vlm/
# eWaSR-Lite trên cùng tập con (mốc so sánh)
python tools/eval_mods.py --pred preds/mods/e7_ewasr_h_lite_c64noeqmax_int8 --mods-root ~/datasets/mods \
    --sequences $(cat configs/mods_vlm_subset.txt) --out-json results/mods_vlm/e7_int8.json
# video so sánh 2×2 (thêm --realtime để mô phỏng độ trễ khi thuyền đang chạy)
python tools/make_compare_video.py --seq kope103-00036460-00039500 \
    --boxes "PaliGemma 2=$D/vlm_paligemma2/boxes.json" "Qwen2.5-VL-3B=$D/vlm_qwen25vl/boxes.json" "OWLv2=$D/vlm_owlv2/boxes.json:0.2" \
    --out results/demo/compare_vlm.mp4
```

## 6. Demo, hình và bài báo

```bash
python tools/predict.py --onnx exports/e7_ewasr_h_lite_384x512.onnx --images <thư mục ảnh> --out preds/demo --vis   # ảnh tô màu
.venv/bin/python tools/make_qual_figure.py                         # hình định tính (paper/figs/qual.png)
third_party/.venv-mods/bin/python tools/make_arch_figure.py        # sơ đồ kiến trúc
third_party/.venv-mods/bin/python tools/make_plots.py              # đồ thị γ₂, đánh đổi, bộ calib
```
Bài báo: xem `paper/CONTEXT.md` (file Word `paper/eej/WaRS_EEJ_manuscript.docx` là bản gốc, sửa trong Word).

## 7. Khi có lỗi

| Hiện tượng | Cách xử lý |
|---|---|
| Kaggle báo `partial` / một run `failed` | mở tab **Logs**, tìm dòng `[<tên run>] LỖI` hoặc `Traceback`; các run khác vẫn có kết quả |
| Compile bị "Killed" | hết RAM: tắt việc khác; chạy bước 2 `--from-quantized` riêng |
| Máy tắt/khởi động lại giữa chừng | chạy lại đúng lệnh: dự đoán/chấm/compile bỏ qua phần đã xong ở mức file (chấm lại phần dở) |
| `ssh` không vào được Pi | Pi chưa bật hoặc khác mạng: `ping raspberrypi.local` |
| Kết quả INT8 lệch nhiều so với FP32 | thử bộ calib khác (`--calib-seed`), luôn chấm bằng MODS chứ không chỉ độ trùng pixel |
