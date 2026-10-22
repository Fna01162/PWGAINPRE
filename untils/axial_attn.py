import math
import torch
import torch.nn as nn
from untils.data import PAD_LEFT, ORI_W


class AxialWAttention2D(nn.Module):
    """沿 W 维（方位角）做 1D self-attention，带周期性位置编码。

    输入/输出: (B, C, H, W)，W=128（含 padding），真实方位角映射到 0..124。
    token 数 = W，attention 矩阵 W×W，计算量极轻。
    """

    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} 必须能被 num_heads={num_heads} 整除"
        self.dim = dim
        self.num_heads = num_heads

        # Pre-norm attention block
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, bias=bias, dropout=0.0,
        )
        self.pos_proj = nn.Linear(2, dim, bias=bias)

        # Pre-norm FFN block
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=bias),
        )

        # 零初始化：训练初期两个分支都不生效
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.channel_gate = nn.Parameter(torch.ones(dim))
        self.alpha_ff = nn.Parameter(torch.zeros(1))
        self.channel_gate_ff = nn.Parameter(torch.ones(dim))

    def _periodic_pos(self, W, device, dtype):
        idx = torch.arange(W, device=device, dtype=torch.long)
        idx = (idx - PAD_LEFT) % ORI_W
        phase = 2 * math.pi * idx.float() / ORI_W
        pos = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.pos_proj(pos.to(dtype))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        residual = x

        # Attention branch: pre-norm → pos → attn → residual
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        x_flat = self.norm1(x_flat)                            # 先 norm
        pos = self._periodic_pos(W, x.device, x.dtype)         # (W, C)
        x_flat = x_flat + pos.unsqueeze(0)                     # 再加 pos
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = residual + self.alpha_attn * self.channel_gate.view(1, -1, 1, 1) * attn_out

        # FFN branch: pre-norm → ff → residual
        x_ff = x.permute(0, 2, 3, 1)                           # (B, H, W, C)
        x_ff = self.norm2(x_ff)                                # 沿 C 归一化
        x_ff = x_ff.permute(0, 3, 1, 2).contiguous()           # (B, C, H, W)
        x = x + self.alpha_ff * self.channel_gate_ff.view(1, -1, 1, 1) * self.ff(x_ff)

        return x


class AxialWAttention3D(nn.Module):
    """5D 版，兼容 (B, C, T, H, W)。对每个 T 独立做 W 方向 attention。"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.inner = AxialWAttention2D(dim, num_heads=num_heads)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(B * T, C, H, W)
        x = self.inner(x)
        x = x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return x
