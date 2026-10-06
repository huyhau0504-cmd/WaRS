import numpy as np

# Chuẩn hóa ImageNet (thang 0–255). Cùng hằng số này được đưa vào model script của Hailo
# (lớp normalization chạy trên chip), nên ảnh đưa vào HEF là RGB uint8 chưa chuẩn hóa.
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
