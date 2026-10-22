# -*- coding: utf-8 -*-
"""U-Net++ 生成器（Zhou et al. 2019 嵌套 U-Net，参考 MrGiovanni/UNetPlusPlus）。

与官方差异（公平性所需）：
  - 所有 3×3 卷积 W 维 circular padding（方位 360° 周期）
  - 输入 6 通道（V×m_eff+噪声 / CNR / SW / Skew / m_eff / rel），输出 1 通道 tanh
  - 深度 4、base_filter 32
判别器：主项目 model.PatchGANDiscriminator（V+hint 2 通道）。

节点约定：X[i][j] = 第 i 层（0=最细分辨率）、第 j 跳深度。
  j=0: X[i,0] = Conv(pool(X[i-1,0]))   （i>0），X[0,0]=Conv(x)
  j>0: X[i,j] = Conv(concat(X[i,j-1], up(X[i+1,j-1])))
输出头接在 X[0][depth]。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from untils.circular_pad import CircularWConv2d  # noqa: E402


def _conv(in_ch, out_ch):
    return nn.Sequential(
        CircularWConv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = _conv(in_ch, out_ch)
        self.c2 = _conv(out_ch, out_ch)

    def forward(self, x):
        return self.c2(self.c1(x))


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, base_filter=32, depth=4):
        super().__init__()
        assert depth >= 2
        self.depth = depth
        bf = base_filter
        chs = [bf * (2 ** i) for i in range(depth + 1)]

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # X[i][0], i=0..depth-1
        self.stem = DoubleConv(in_channels, chs[0])
        self.down = nn.ModuleList(
            DoubleConv(chs[i], chs[i + 1]) for i in range(depth - 1))
        # bottleneck X[depth][0]
        self.bottleneck = DoubleConv(chs[depth - 1], chs[depth])
        # X[i][j], j=1..depth-i, i=0..depth-1（顺序与 forward 的 j 外层循环一致）
        self.nest = nn.ModuleList()
        for j in range(1, depth + 1):
            for i in range(depth - j + 1):
                self.nest.append(DoubleConv(chs[i] + chs[i + 1], chs[i]))

        self.head = nn.Sequential(
            CircularWConv2d(chs[0], out_channels, 1),
            nn.Tanh(),
        )

    def _up_to(self, x, ref):
        y = self.up(x)
        if y.shape[-2:] != ref.shape[-2:]:
            y = nn.functional.interpolate(y, size=ref.shape[-2:],
                                          mode="bilinear", align_corners=False)
        return y

    def forward(self, x):
        B, _, H, W = x.shape
        depth = self.depth
        # X[i][j]: i=0..depth, 仅 j <= depth-i 有效
        X = [[None] * (depth + 1) for _ in range(depth + 1)]

        X[0][0] = self.stem(x)
        for i in range(1, depth):
            X[i][0] = self.down[i - 1](self.pool(X[i - 1][0]))
        X[depth][0] = self.bottleneck(self.pool(X[depth - 1][0]))

        k = 0
        for j in range(1, depth + 1):
            for i in range(depth - j + 1):
                up_n = self._up_to(X[i + 1][j - 1], X[i][j - 1])
                X[i][j] = self.nest[k](torch.cat([X[i][j - 1], up_n], dim=1))
                k += 1

        out = self.head(X[0][depth])
        if out.shape[-2:] != (H, W):
            out = nn.functional.interpolate(out, size=(H, W),
                                            mode="bilinear", align_corners=False)
        return out


def build_unetpp_cgan(base_filter=32, depth=4):
    """返回 (G, D)。"""
    from model import PatchGANDiscriminator
    G = UNetPlusPlus(in_channels=6, out_channels=1,
                     base_filter=base_filter, depth=depth)
    D = PatchGANDiscriminator(in_channels=2, base_filter=24)
    return G, D
