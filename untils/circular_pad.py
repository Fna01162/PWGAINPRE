"""W 方向（方位角）circular padding，H/T 方向 zero padding 的卷积包装。

雷达 PPI 数据中：
  W 轴 = 方位角（0°~360° 周期性），需要 circular padding
  H 轴 = 距离门（近场~远场，非周期），保持 zero padding
  T 轴 = 时间步（T=1），保持 zero padding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularWConv3d(nn.Module):
    """3D 卷积：W 维 circular，H/T 维 zero。适用于 kernel=3, padding=1。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        assert kernel_size == 3, "目前只支持 3x3x3 kernel"
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0, groups=groups, bias=bias)

    def forward(self, x):
        # x: (B, C, T, H, W)
        p = 1
        # W 方向 circular
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
        # T、H 方向 zero（F.pad 维度顺序: W, H, T）
        x = F.pad(x, (0, 0, p, p, p, p), mode="constant", value=0.0)
        return self.conv(x)


class CircularWConv2d(nn.Module):
    """2D 卷积：W 维 circular，H 维 zero。适用于 Discriminator。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.p = padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0, bias=bias)

    def forward(self, x):
        # x: (B, C, H, W)
        p = self.p
        if p > 0:
            x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
            x = F.pad(x, (0, 0, p, p), mode="constant", value=0.0)
        return self.conv(x)
