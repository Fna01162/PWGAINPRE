# -*- coding: utf-8 -*-
"""DSA-UNet 生成器（Gong et al., Remote Sens. 2023, 15, 4568）。

官方实现：compare_baselines/third_party/Radar-Completion/models/dsa_unet.py
结构（~18.17M 参数）：
  in_conv 1×1 (in→32)
  4× 下采样 DoubleConv: 32→64→128→256→512 (k3 s2 p1)
  4× 空洞卷积 (512ch, dilation 1/2/4/8)
  4× 上采样 DoubleDeconv: cat 后 1024→256→128→64→32 (ConvTranspose k3 s2 p1 op1)
  前两个上采样输出接 SelfAttention (channel/8 query-key, gamma 残差)
  out 1×1 (64→1) + Sigmoid

与官方的公平性改动（方案要求）：
  - 所有 3×3 卷积 W 维 circular padding（官方用 ±20 列显式方位 padding，此处改为
    周期卷积，等价消除 0°/360° 接缝伪影且不改变张量宽度）
  - 输入 6 通道（V×m_eff+噪声 / CNR / SW / Skew / m_eff / rel），输出 1 通道
  - ConvTranspose2d 保持零填充（方案只约束 Conv2d 的 W 维 padding）
判别器：主项目 model.PatchGANDiscriminator（V+hint 2 通道）。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # noqa: E402


class CircularWConv2dD(nn.Module):
    """W 维 circular、H 维 zero padding 的 2D 卷积（支持 dilation）。

    untils.circular_pad.CircularWConv2d 不支持 dilation，且主项目文件禁改，
    故在本模块内实现等价版本。
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True):
        super().__init__()
        assert kernel_size == 3 and dilation >= 1
        self.p = padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0, dilation=dilation, bias=bias)

    def forward(self, x):
        p = self.p
        if p > 0:
            # 与 PyTorch Conv2d(padding=p, dilation=d) 语义一致：两侧各扩 p 个元素。
            # 官方 dilated conv 恒有 p == d，扩 p 列即可覆盖 dilation=2d 的感受野。
            x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
            x = F.pad(x, (0, 0, p, p), mode="constant", value=0.0)
        return self.conv(x)


def _cconv(in_ch, out_ch, k=3, s=1, p=1, d=1):
    return CircularWConv2dD(in_ch, out_ch, k, stride=s, padding=p, dilation=d)


class UNetDoubleConv2d(nn.Module):
    """官方 DoubleConv 的 circular 版：Conv+BN+ReLU ×2。"""

    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            _cconv(in_ch, out_ch, k, s, p, d),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            _cconv(out_ch, out_ch, k, 1, p, d),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDoubleDeconv2d(nn.Module):
    """官方 DoubleDeconv：ConvT+BN+ReLU 后接 circular Conv+BN+ReLU。"""

    def __init__(self, in_ch, out_ch, k=3, s=2, p=1, op=1):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, k, stride=s, padding=p,
                               output_padding=op),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            _cconv(out_ch, out_ch, k, 1, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        y = self.deconv(x)
        if y.shape[-2:] != skip.shape[-2:]:
            y = F.interpolate(y, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(y)


class SelfAttention(nn.Module):
    """官方 SelfAttention 原样复用（无空间卷积，无需 circular）。"""

    def __init__(self, in_ch):
        super().__init__()
        self.query_conv = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.key_conv = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.value_conv = nn.Conv2d(in_ch, in_ch, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query_conv(x).view(B, -1, H * W).transpose(1, 2)
        k = self.key_conv(x).view(B, -1, H * W)
        v = self.value_conv(x).view(B, -1, H * W)
        attn = self.softmax(torch.bmm(q, k))
        out = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        return self.gamma * out + x


class DSAUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.down1 = UNetDoubleConv2d(32, 64)
        self.down2 = UNetDoubleConv2d(64, 128)
        self.down3 = UNetDoubleConv2d(128, 256)
        self.down4 = UNetDoubleConv2d(256, 512)
        self.dilated_conv1 = nn.Sequential(
            _cconv(512, 512, p=1, d=1), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.dilated_conv2 = nn.Sequential(
            _cconv(512, 512, p=2, d=2), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.dilated_conv3 = nn.Sequential(
            _cconv(512, 512, p=4, d=4), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.dilated_conv4 = nn.Sequential(
            _cconv(512, 512, p=8, d=8), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.up4 = UNetDoubleDeconv2d(1024, 256)
        self.attn4 = SelfAttention(256)
        self.up3 = UNetDoubleDeconv2d(512, 128)
        self.attn3 = SelfAttention(128)
        self.up2 = UNetDoubleDeconv2d(256, 64)
        self.up1 = UNetDoubleDeconv2d(128, 32)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h1 = self.in_conv(x)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h5 = self.down4(h4)
        hd = self.dilated_conv1(h5)
        hd = self.dilated_conv2(hd)
        hd = self.dilated_conv3(hd)
        hd = self.dilated_conv4(hd)
        h4p = self.attn4(self.up4(torch.cat([hd, h5], dim=1), h4))
        h3p = self.attn3(self.up3(torch.cat([h4p, h4], dim=1), h3))
        h2p = self.up2(torch.cat([h3p, h3], dim=1), h2)
        h1p = self.up1(torch.cat([h2p, h2], dim=1), h1)
        out = self.out_conv(torch.cat([h1p, h1], dim=1))
        return self.act(out)


def build_dsa_unet():
    """返回 (G, D)。"""
    from model import PatchGANDiscriminator
    G = DSAUNet(in_channels=6, out_channels=1)
    D = PatchGANDiscriminator(in_channels=2, base_filter=24)
    return G, D
