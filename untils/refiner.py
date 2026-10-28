# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from untils.circular_pad import CircularWConv3d
from untils.NAF_Block import NAFBlock3D
from untils.data import ORI_H, ORI_W, pad_to
from untils.wavelet import wavelet_features, wavelet_feature_channels


def _pool3d(x):
    kt = 2 if x.shape[2] >= 2 else 1
    return F.avg_pool3d(
        x,
        kernel_size=(kt, 2, 2),
        stride=(kt, 2, 2),
    )


class ResidualRefiner(nn.Module):
    """小波 + 物理条件引导的残差细化网络。"""

    def __init__(self, in_channels=13, base_filter=16):
        super().__init__()
        bf = int(base_filter)

        def conv3(cin, cout):
            return nn.Sequential(
                CircularWConv3d(cin, cout, 3),
                nn.GroupNorm(max(1, min(8, cout)), cout),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.enc1 = conv3(in_channels, bf)
        self.enc2 = conv3(bf, bf * 2)
        self.bott = NAFBlock3D(bf * 2, bf * 2)
        self.up = CircularWConv3d(bf * 2, bf, 3)

        self.dec = nn.Sequential(
            CircularWConv3d(bf * 2, bf, 3),
            nn.GroupNorm(max(1, min(8, bf)), bf),
            nn.LeakyReLU(0.1, inplace=True),
            CircularWConv3d(bf, 2, 3),
        )

        self._init_output()

    def _init_output(self):
        """让 Refiner 初始阶段接近 ΔV ≈ 0，避免破坏已经训练好的 GAIN_NAF。"""
        last = self.dec[-1]
        if isinstance(last, CircularWConv3d):
            conv = getattr(last, "conv", None)
            if conv is None:
                return
            nn.init.zeros_(conv.weight)
            if getattr(conv, "bias", None) is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(_pool3d(e1))
        b = self.bott(e2)
        d = self.up(b)
        d = F.interpolate(d, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d = torch.cat([d, e1], dim=1)
        out = self.dec(d)

        raw = out[:, :1]
        gate_logits = out[:, 1:2]
        gate = torch.sigmoid(gate_logits)
        return raw, gate


_GEOMETRY_MAPS = None


def build_geometry_maps(device=None, dtype=torch.float32):
    """构造 PPI 几何条件：Range, sin(theta), cos(theta)"""
    global _GEOMETRY_MAPS
    if _GEOMETRY_MAPS is None:
        dist = 660.0 + 30.0 * np.arange(ORI_H, dtype=np.float32)
        dist_norm = dist / dist[-1]
        az_rad = np.linspace(0.0, 360.0, ORI_W, endpoint=False, dtype=np.float32) * np.pi / 180.0
        rmap = np.tile(dist_norm[:, None], (1, ORI_W))
        smap = np.tile(np.sin(az_rad)[None, :], (ORI_H, 1))
        cmap = np.tile(np.cos(az_rad)[None, :], (ORI_H, 1))

        def to_padded(x):
            x = x.astype(np.float32, copy=False)
            return pad_to(x[None])[0]

        _GEOMETRY_MAPS = (
            torch.from_numpy(to_padded(rmap)),
            torch.from_numpy(to_padded(smap)),
            torch.from_numpy(to_padded(cmap)),
        )
    maps = tuple(m.to(device=device, dtype=dtype) for m in _GEOMETRY_MAPS)
    return maps


def _broadcast_map(x, batch_size):
    """[H,W] → [B,1,H,W]"""
    return x.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, x.shape[-2], x.shape[-1])


def build_refiner_input(v_gain, x, given_mask, rel_mask, cfg, device, use_wavelet=True):
    """构造 Residual Refiner 输入。"""
    target_t = int(getattr(cfg, "target_idx", 0))
    B = v_gain.shape[0]
    rmap, smap, cmap = build_geometry_maps(device=device, dtype=v_gain.dtype)

    parts = [v_gain]
    phys_ch = int(getattr(cfg, "phys_channels", 4))

    for c in range(1, phys_ch):
        x_slice = x[:, c:c + 1, target_t]
        parts.append(x_slice)

    parts.extend([
        _broadcast_map(rmap, B),
        _broadcast_map(smap, B),
        _broadcast_map(cmap, B),
    ])

    given_t = given_mask[:, target_t:target_t + 1] if given_mask.ndim == 5 else given_mask
    rel_t = rel_mask[:, target_t:target_t + 1] if rel_mask.ndim == 5 else rel_mask

    # [FIX] 确保掩码是单通道：如果多通道（每个物理通道一个掩码），只取目标通道（V）的掩码
    # Refiner 只补全目标通道（target_t，默认 0 = V 速度场），所以掩码应该对齐
    if given_t.ndim == 4 and given_t.shape[1] > 1:
        given_t = given_t[:, target_t:target_t + 1, :, :]
    if rel_t.ndim == 4 and rel_t.shape[1] > 1:
        rel_t = rel_t[:, target_t:target_t + 1, :, :]

    parts.extend([given_t, rel_t])

    if use_wavelet:
        levels = int(getattr(cfg, "n_wavelet_levels", 1))
        wf = wavelet_features(v_gain, levels=levels)
        parts.append(wf)

    inp = torch.cat(parts, dim=1)

    return inp.unsqueeze(2)


def clamp_delta(delta, max_delta=0.5):
    """限制 Refiner 残差幅度，防止外推时产生过冲伪影。"""
    if max_delta is None:
        return delta
    max_delta = float(max_delta)
    if max_delta <= 0:
        return delta
    return torch.clamp(delta, -max_delta, max_delta)


def refine_delta_and_gate(R, v_gain, x, given_mask, rel_mask, cfg, device, use_wavelet=True):
    """Refiner 前向推理。"""
    inp = build_refiner_input(
        v_gain=v_gain, x=x, given_mask=given_mask, rel_mask=rel_mask,
        cfg=cfg, device=device, use_wavelet=use_wavelet,
    )
    raw, gate = R(inp)
    delta = (gate * raw)[:, :, 0]

    # [核心修复] 启用 Delta 截断，抑制低信噪比边缘的外推伪影
    delta = clamp_delta(delta, getattr(cfg, "refiner_max_delta", 0.30))

    gate = gate[:, :, 0]
    return delta, gate


def merge_v_final(v_gain, delta, known_mask):
    """将 Refiner 残差作用到未知区域。"""
    known_mask = known_mask.float()
    if known_mask.ndim == 3:
        known_mask = known_mask.unsqueeze(1)
    return known_mask * v_gain + (1.0 - known_mask) * (v_gain + delta)


def apply_refiner(R, v_gain, x, given_mask, rel_mask, valid_mask, cfg, device):
    """推理阶段调用 Refiner。"""
    delta, gate = refine_delta_and_gate(
        R=R, v_gain=v_gain, x=x, given_mask=given_mask, rel_mask=rel_mask,
        cfg=cfg, device=device, use_wavelet=getattr(cfg, "use_wavelet", True),
    )

    t = getattr(cfg, "target_idx", 0)
    valid_t = valid_mask[:, t:t + 1] if valid_mask.ndim == 5 else valid_mask
    valid_t = valid_t.unsqueeze(1) if valid_t.ndim == 3 else valid_t
    # [FIX] 处理多通道掩码情况（每个物理通道一个掩码）
    if valid_t.ndim == 4 and valid_t.shape[1] > 1:
        valid_t = valid_t[:, t:t + 1, :, :]

    given_t = given_mask[:, t:t + 1] if given_mask.ndim == 5 else given_mask
    given_t = given_t.unsqueeze(1) if given_t.ndim == 3 else given_t
    # [FIX] 处理多通道掩码情况
    if given_t.ndim == 4 and given_t.shape[1] > 1:
        given_t = given_t[:, t:t + 1, :, :]

    rel_t = rel_mask[:, t:t + 1] if rel_mask.ndim == 5 else rel_mask
    rel_t = rel_t.unsqueeze(1) if rel_t.ndim == 3 else rel_t
    # [FIX] 处理多通道掩码情况
    if rel_t.ndim == 4 and rel_t.shape[1] > 1:
        rel_t = rel_t[:, t:t + 1, :, :]

    known = given_t * rel_t * valid_t
    v_final = merge_v_final(v_gain, delta, known)
    return v_final