from .ewasr import EWaSR

# Các biến thể dùng trong đề tài. Chữ cái mixer: C = ARM (kênh), S = spatial mean/max (gốc),
# D = depthwise spatial gate (eWaSR-H). Thêm biến thể mới cho ablation tại đây.
MODELS = {
    "ewasr": dict(mixer="CCCCSS", enricher="SS"),    # eWaSR gốc, nạp được trọng số chính thức
    "ewasr_h": dict(mixer="CCCCDD", enricher="DD"),  # eWaSR-H thân thiện Hailo
    "ewasr_c": dict(mixer="CCCCCC", enricher="CC"),  # ablation: chỉ attention theo kênh
    "ewasr_h_p": dict(mixer="CCCCDD", enricher="DD", project=True),  # eWaSR-H, decoder giảm kênh
}


def get_model(name, pretrained_backbone=True, **overrides):
    if name not in MODELS:
        raise ValueError(f"Mô hình không hỗ trợ: {name}. Có: {list(MODELS)}")
    return EWaSR(**{**MODELS[name], **overrides}, pretrained_backbone=pretrained_backbone)


def load_weights(model, path, strict=True):
    """Nạp state_dict từ file .pth (trọng số eWaSR chính thức hoặc checkpoint của repo này).

    strict=False: chỉ nạp các tham số trùng tên & kích thước (vd. khởi tạo eWaSR-H từ eWaSR).
    Trả về danh sách khóa bị bỏ qua.
    """
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    if strict:
        model.load_state_dict(sd)
        return []
    own = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return sorted(set(own) - set(keep))
