# Adapted from eWaSR (https://github.com/tersekmatija/eWaSR), Apache-2.0.
# Tên module/tham số được giữ nguyên để nạp được trọng số eWaSR chính thức.
import torch
import torch.nn.functional as F
from torch import nn


def resize_like(x, ref, mode):
    """Resize `x` về kích thước không gian của `ref`.

    Khi tỉ lệ là số nguyên thì dùng `scale_factor` để ONNX sinh Resize với scale hằng —
    dạng mà Hailo DFC xử lý tốt hơn so với Resize theo `sizes`.
    """
    # int(): shape thành hằng số khi trace ONNX (mô hình xuất với shape tĩnh)
    (h, w), (H, W) = map(int, x.shape[-2:]), map(int, ref.shape[-2:])
    if (h, w) == (H, W):
        return x
    kw = {} if mode == "nearest" else {"align_corners": False}
    if H % h == 0 and W % w == 0:
        return F.interpolate(x, scale_factor=(H // h, W // w), mode=mode, **kw)
    return F.interpolate(x, size=(H, W), mode=mode, **kw)


class AttentionRefinementModule(nn.Module):
    """Attention theo kênh: GAP → conv 1×1 → BN → sigmoid."""

    def __init__(self, in_channels, last_arm=False):
        super().__init__()
        self.last_arm = last_arm
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.sigmoid(self.bn1(self.conv1(self.global_pool(x))))
        out = weights * x
        if self.last_arm:
            out = self.global_pool(out) * out
        return out


class PyramidPoolAgg(nn.Module):
    """Gom các mức đặc trưng về cùng độ phân giải (1/stride của mức thô nhất) rồi nối theo kênh."""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        H, W = map(int, inputs[0].shape[-2:])
        H, W = (H - 1) // self.stride + 1, (W - 1) // self.stride + 1
        return torch.cat([_pool_to(inp, H, W) for inp in inputs], dim=1)


def _pool_to(x, H, W):
    h, w = map(int, x.shape[-2:])
    if h % H == 0 and w % W == 0:
        # Tương đương adaptive pooling nhưng xuất ra ONNX AveragePool có kernel cố định
        k = (h // H, w // W)
        return F.avg_pool2d(x, kernel_size=k, stride=k)
    return F.adaptive_avg_pool2d(x, (H, W))


class SIM(nn.Module):
    """Semantic Injection Module (TopFormer): trộn đặc trưng cục bộ với token toàn cục."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv_lc = nn.Conv2d(ch_in, ch_out, 1)
        self.bn_lc = nn.BatchNorm2d(ch_out)
        self.conv_gc = nn.Conv2d(ch_in, ch_out, 1)
        self.bn_gc = nn.BatchNorm2d(ch_out)
        self.sigmoid_gc = nn.Sigmoid()
        self.conv_gc1 = nn.Conv2d(ch_in, ch_out, 1)
        self.bn_gc1 = nn.BatchNorm2d(ch_out)

    def forward(self, lx, gx):
        gx = resize_like(gx, lx, "nearest")
        lx = self.bn_lc(self.conv_lc(lx))
        multi = self.sigmoid_gc(self.bn_gc(self.conv_gc(gx)))
        gx = self.bn_gc1(self.conv_gc1(gx))
        return lx * multi + gx


class SegHead(nn.Module):
    def __init__(self, ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU6()
        self.conv1 = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        return self.conv1(self.relu(self.bn(self.conv(x))))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# --- Token mixers -----------------------------------------------------------------------------
# C: ARMMixer (gốc)          — attention theo kênh
# S: SpatialAttentionMixer (gốc) — attention không gian từ mean/max theo trục kênh
# D: DWSpatialMixer (eWaSR-H) — cổng không gian bằng depthwise conv, không reduce theo trục kênh


class ARMMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.arm = AttentionRefinementModule(in_channels=dim, last_arm=False)

    def forward(self, x):
        return self.arm(x)


class SpatialAttentionMixer(nn.Module):
    def __init__(self, dim=None, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(att) * x


class DWSpatialMixer(nn.Module):
    """Thay thế thân thiện phần cứng cho SpatialAttentionMixer.

    Chỉ dùng depthwise conv 3×3, conv 1×1, BN và sigmoid — tránh ReduceMax/ReduceMean theo
    trục kênh, là các phép mà bộ tăng tốc dataflow thường hỗ trợ kém hoặc chạy trên CPU.
    """

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.pw(self.bn(self.dw(x)))) * x


TOKEN_MIXERS = {"C": ARMMixer, "S": SpatialAttentionMixer, "D": DWSpatialMixer}


class MetaFormerBlock(nn.Module):
    """use_mlp=False: bỏ hẳn nhánh MLP (không tạo norm2/mlp/layer_scale_2) — biến thể mixer-lite.
    Tên các tham số còn lại giữ như bản gốc để nạp trọng số eWaSR bằng strict=False."""

    def __init__(self, dim, token_mixer, mlp_ratio=4.0, layer_scale_init_value=1e-5, use_mlp=True):
        super().__init__()
        self.use_mlp = use_mlp
        self.norm1 = nn.BatchNorm2d(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        if use_mlp:
            self.norm2 = nn.BatchNorm2d(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        x = x + self.layer_scale_1[:, None, None] * self.token_mixer(self.norm1(x))
        if self.use_mlp:
            x = x + self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x))
        return x
