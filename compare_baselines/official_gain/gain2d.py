# -*- coding: utf-8 -*-
"""官方 GAIN 的 2D 卷积改造（jsyoon0823/GAIN 参考，见 third_party/GAIN/gain.py）。

官方为表格数据 MLP（G/D 各 3 层，sigmoid 输出，逐特征）。本改造：
  - G：3 级下采样卷积编码器 + 上采样解码器（circular W padding），
       输入 [B,6,H,W]（V×m_eff+噪声 / CNR / SW / Skew / m_eff / rel），
       输出 [B,1,H,W] tanh（归一化 V ∈ [-1,1]）。
  - D：PatchGAN（model.PatchGANDiscriminator，输入 V+hint 2 通道）。
  - 损失严格按官方：D_loss=BCE(D(Hat_X,H), M)；G_loss=adv+alpha*MSE（alpha=10）。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from untils.circular_pad import CircularWConv2d  # noqa: E402


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            CircularWConv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Gain2DGenerator(nn.Module):
    """GAIN Generator（2D 卷积版）。in=6 -> out=1，tanh。"""

    def __init__(self, in_channels=6, base_filter=32):
        super().__init__()
        bf = base_filter
        self.enc1 = ConvBlock(in_channels, bf)
        self.enc2 = ConvBlock(bf, bf * 2)
        self.enc3 = ConvBlock(bf * 2, bf * 4)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(bf * 4 + bf * 2, bf * 2)
        self.dec2 = ConvBlock(bf * 2 + bf, bf)
        self.dec1 = nn.Sequential(
            CircularWConv2d(bf, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _up_to(self, x, ref):
        y = self.up(x)
        if y.shape[-2:] != ref.shape[-2:]:
            y = nn.functional.interpolate(y, size=ref.shape[-2:],
                                          mode="bilinear", align_corners=False)
        return y

    def forward(self, x):
        e1 = self.enc1(x)              # [B,bf,H,W]
        e2 = self.enc2(self.pool(e1))  # [B,2bf,H/2,W/2]
        e3 = self.enc3(self.pool(e2))  # [B,4bf,H/4,W/4]
        d3 = self.dec3(torch.cat([self._up_to(e3, e2), e2], dim=1))
        d2 = self.dec2(torch.cat([self._up_to(d3, e1), e1], dim=1))
        return self.dec1(d2)


def build_gain2d(base_filter=32):
    """返回 (G, D)。D 用主项目 PatchGANDiscriminator（2 通道输入）。"""
    from model import PatchGANDiscriminator
    G = Gain2DGenerator(in_channels=6, base_filter=base_filter)
    D = PatchGANDiscriminator(in_channels=2, base_filter=24)
    return G, D
