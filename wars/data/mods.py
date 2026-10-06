from pathlib import Path


def list_mods_frames(mods_root):
    """Liệt kê khung hình MODS cần dự đoán: `sequences/<seq>/frames/<name>.jpg`.

    Giống WaSR, danh sách lấy theo thư mục `imus/` (mỗi khung trái có một IMU mask).
    Trả về list (seq, đường dẫn ảnh).
    """
    out = []
    for seq_dir in sorted((Path(mods_root) / "sequences").iterdir()):
        for imu in sorted((seq_dir / "imus").glob("*.png")):
            out.append((seq_dir.name, seq_dir / "frames" / f"{imu.stem}.jpg"))
    return out
