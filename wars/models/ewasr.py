# Adapted from eWaSR (https://github.com/tersekmatija/eWaSR), Apache-2.0.
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter

from .layers import SIM, TOKEN_MIXERS, MetaFormerBlock, PyramidPoolAgg, SegHead, resize_like


class EWaSRDecoder(nn.Module):
    def __init__(self, num_classes=3, ch=512, ch_sim=256, mixer="CCCCSS", enricher="SS", project=False,
                 mixer_mlp=None):
        super().__init__()
        self.ch = [ch, ch // 2, ch // 4, ch // 8]
        self.project = project
        if project:
            # Giảm một nửa số kênh trước feature mixer → MetaFormer nhẹ ~4 lần
            self.convs_project1 = nn.Conv2d(self.ch[0], self.ch[0] // 2, 1)
            self.convs_project2 = nn.Conv2d(self.ch[1], self.ch[1] // 2, 1)
            self.convs_project3 = nn.Conv2d(self.ch[2], self.ch[2] // 2, 1)
            self.convs_project4 = nn.Conv2d(self.ch[3], self.ch[3] // 2, 1)
            self.ch = [c // 2 for c in self.ch]
        self.pool_feats = PyramidPoolAgg(2)
        self.metaformers = nn.Sequential(
            *[MetaFormerBlock(sum(self.ch), token_mixer=TOKEN_MIXERS[m], use_mlp=(mixer_mlp or "1" * len(mixer))[i] == "1")
              for i, m in enumerate(mixer)]
        ) if mixer else None
        self.metaformers_skip2 = nn.Sequential(
            *[MetaFormerBlock(self.ch[2], token_mixer=TOKEN_MIXERS[m]) for m in enricher]
        ) if enricher else None
        self.sim1 = SIM(self.ch[0], ch_sim)
        self.sim2 = SIM(self.ch[1], ch_sim)
        self.sim3 = SIM(self.ch[2], ch_sim)
        self.sim4 = SIM(self.ch[3], ch_sim)
        self.seg_head = SegHead(ch_sim, num_classes)

    def forward(self, x, aux, skip2, skip1):
        if self.project:
            x = self.convs_project1(x)
            aux = self.convs_project2(aux)
            skip2 = self.convs_project3(skip2)
            skip1 = self.convs_project4(skip1)
        tokens = self.pool_feats([x, aux, skip2, skip1])
        if self.metaformers is not None:
            tokens = self.metaformers(tokens)
        if self.metaformers_skip2 is not None:
            skip2 = self.metaformers_skip2(skip2)

        c = [0, *[sum(self.ch[: i + 1]) for i in range(4)]]
        f1 = self.sim1(x, tokens[:, c[0]:c[1]])
        f2 = self.sim2(aux, tokens[:, c[1]:c[2]])
        f3 = self.sim3(skip2, tokens[:, c[2]:c[3]])
        f4 = self.sim4(skip1, tokens[:, c[3]:c[4]])

        x = (resize_like(f1, f4, "bilinear") + resize_like(f2, f4, "bilinear")
             + resize_like(f3, f4, "bilinear") + f4)
        return self.seg_head(x)


class EWaSR(nn.Module):
    """eWaSR không IMU. Đầu vào: ảnh đã chuẩn hóa ImageNet (B,3,H,W).

    Đầu ra: logits 3 lớp (0 vật cản, 1 nước, 2 trời) ở độ phân giải H/4 × W/4.
    Upsample về kích thước ảnh được làm ở hậu xử lý (ngoài mô hình) để giảm tải cho Hailo.
    Với `return_aux=True` trả thêm đặc trưng layer3 dùng cho separation loss khi huấn luyện.
    """

    def __init__(self, num_classes=3, mixer="CCCCSS", enricher="SS", project=False, pretrained_backbone=True,
                 mixer_mlp=None):
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None)
        self.backbone = IntermediateLayerGetter(
            bb, return_layers={"layer1": "skip1", "layer2": "skip2", "layer3": "aux", "layer4": "out"}
        )
        self.decoder = EWaSRDecoder(num_classes=num_classes, ch=512, mixer=mixer, enricher=enricher,
                                    project=project, mixer_mlp=mixer_mlp)

    def forward(self, image, return_aux=False):
        f = self.backbone(image)
        logits = self.decoder(f["out"], f["aux"], f["skip2"], f["skip1"])
        return (logits, f["aux"]) if return_aux else logits
