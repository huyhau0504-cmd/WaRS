# Adapted from WaSR / eWaSR (Apache-2.0).
import torch
import torch.nn.functional as F


def focal_loss(logits, labels, gamma=2.0):
    """Focal loss với nhãn one-hot mềm (B,3,H,W). Pixel bỏ qua có one-hot toàn 0 nên không đóng góp."""
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    p = torch.softmax(logits, 1)
    fl = -labels * torch.log(p + 1e-9) * (1.0 - p) ** gamma
    return fl.sum(1).mean()


def water_obstacle_separation_loss(features, labels, eps=0.01, min_samples=5):
    """Loss tách nước–vật cản của WaSR: kéo đặc trưng nước co cụm, đẩy đặc trưng vật cản ra xa."""
    labels = F.interpolate(labels, size=features.shape[-2:], mode="area")
    m_water = labels[:, 1:2]
    m_obst = labels[:, 0:1]
    n_water = m_water.sum((0, 2, 3), keepdim=True).clamp(min=1.0)
    n_obst = m_obst.sum((0, 2, 3), keepdim=True)
    if n_obst.squeeze() < min_samples or n_water.squeeze() < min_samples:
        return features.new_zeros(())

    water = m_water * features
    obst = m_obst * features
    mean_water = water.sum((0, 2, 3), keepdim=True) / n_water
    var_water = (water - mean_water * m_water).pow(2).sum((0, 2, 3), keepdim=True) / n_water
    diff = (obst - mean_water * m_obst).pow(2).sum((0, 2, 3), keepdim=True)
    return (n_obst * var_water / (diff + eps)).mean()
