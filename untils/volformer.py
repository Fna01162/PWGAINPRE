import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecayPos1d(nn.Module):
    """通道维衰减位置先验，来自VolFormer官方"""

    def __init__(self, embed_dim, num_heads, initial_value=2, heads_range=4):
        super().__init__()
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('decay', decay)

    def forward(self, c_len):
        index = torch.arange(c_len, device=self.decay.device)
        mask = (index[:, None] - index[None, :]).abs()  # (c, c)
        return mask * self.decay[:, None, None]  # (n_heads, c, c)


class VolSelfAttention3D(nn.Module):
    """VolFormer体积自注意力的3D适配版，输入输出均为(B,C,T,H,W)"""

    def __init__(self, dim, num_heads=4, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 空间分支QKV
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # 通道分支QKV（用1x1conv实现）
        self.qkv_C = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv_C = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False)
        self.proj_C = nn.Conv2d(dim, dim, kernel_size=1)

        self.realPos = DecayPos1d(64, num_heads, 2, 4)

        # 空间注意力引导模块: 保留原版 3x3 卷积+BN, 修两个 bug
        #   bug1: AdaptiveAvgPool2d(1) 把 N 也池掉了 -> 改成只池 keys 维, 保留 N
        #   bug2: 最后一层写死输出 64 -> 改成输出 1 通道, 任意 N 都对
        self.Gao_spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),   # (B, heads, N, 1)
            nn.Conv2d(num_heads, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3, 1, 1),        # 输出 1 通道门控
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, C, T, H, W) → 时间维参与注意力，T*H*W展平为空间token（真正的三维立方体）
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)  # (B, N, C), N=T*H*W
        B_, N, C = x.shape
        hh, ww = H, W

        # ===== 空间分支 =====
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, heads, N, N)
        attn = self.softmax(attn)
        x1 = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x1 = self.proj(x1)

        # ===== 通道分支 =====
        x_s = x.reshape(B_, hh, ww, C).permute(0, 3, 1, 2)  # (B_, C, H, W)
        qkv_c = self.qkv_dwconv_C(self.qkv_C(x_s))
        q_c, k_c, v_c = qkv_c.chunk(3, dim=1)
        # (B_, heads, c, N)
        q_c = q_c.reshape(B_, self.num_heads, C // self.num_heads, N)
        k_c = k_c.reshape(B_, self.num_heads, C // self.num_heads, N)
        v_c = v_c.reshape(B_, self.num_heads, C // self.num_heads, N)
        q_c = F.normalize(q_c, dim=-1)
        k_c = F.normalize(k_c, dim=-1)
        realPos = self.realPos(C // self.num_heads)  # (heads, c, c)
        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature + realPos.unsqueeze(0)
        attn_c = attn_c.softmax(dim=-1)
        x2 = (attn_c @ v_c)  # (B_, heads, c, N)
        x2 = x2.reshape(B_, C, N).reshape(B_, C, hh, ww)
        x2 = self.proj_C(x2)
        x2 = x2.flatten(2).transpose(1, 2)  # (B_, N, C)

        # ===== 空间注意力引导通道特征 =====
        attn_s = self.Gao_spatial(attn)          # (B_, 1, N, 1)
        attn_s = attn_s.reshape(B_, N, 1)        # (B_, N, 1)
        x4 = attn_s * x2

        out = x1 + x2 + x4
        # 恢复时间维
        out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        return out