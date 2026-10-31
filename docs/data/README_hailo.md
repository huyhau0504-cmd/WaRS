# Số liệu Hailo-8L đóng băng (31/10/2026, Q-016)

Giao thức chính thức: DFC 3.34, opt level 0, **64 ảnh calib (MaSTr1325 train, `--calib-seed 0`), tắt cross-layer equalization (`--no-equalization`), compiler `max`**; chọn equalization theo trung bình 3 seed trên USV-VN valid. HailoRT 4.23, Pi 5 + Hailo-8L, 384×512.

- `hailo_main_table.csv` — bảng chính: context, FPS phần cứng (`hailortcli benchmark -t 15`), HW latency, thông lượng đầu-cuối (`pi/run.py` 1 làm nóng + 3 lượt, trung vị/min/max; đọc thẻ và `--preload`), md5 HEF. HEF nằm trong `hailo/build/` (không commit).
- `e2e/<hef>_{disk,pre}.json` — `timing.json` gốc (nhiệt độ/throttle mỗi lượt, governor).
- `int8_agreement_seeds.csv` — % pixel INT8 trùng FP32 (ONNX) theo split × backend (emulator/Pi) × mô hình × eq × seed calib.
- `limvals_growth.csv`, `activation_stats.csv`, `layer_scale.csv` — phân tích C2 (Q-012/Q-013).
Nguồn lập luận: `.coord/INBOX.md` Q-011 → Q-016.
