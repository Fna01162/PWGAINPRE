# -*- coding: utf-8 -*-
"""深度学习补全基线的精简实现。

补丁式（局部窗口，变量 = 补丁内格点，序列 = 时间帧）：
    BRITS  —— 双向 RITS，带时间衰减与特征回归
    SAITS  —— 两层自注意力
网格原生（整幅 PPI，时间帧序列）：
    MemoryUNet —— 偏置卷积 / 普通卷积编码 + 瓶颈双向 ConvLSTM + 解码
        partial=True  → HESS（偏置卷积 + 记忆）
        partial=False → BiConvLSTM-UNet（普通卷积，不喂掩码）
另外：
    Refine3d —— 3D 残差细化，用于把一阶段 GAN 结果做二阶段细化（论文 Refiner 的精简版）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from predictor import ConvLSTMCell


# ============================ BRITS ============================

class FeatureRegression(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n, n))
        self.b = nn.Parameter(torch.zeros(n))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        return x @ self.w + self.b


class TemporalDecay(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.map = nn.Linear(n, h)

    def forward(self, delta):
        return torch.exp(-F.relu(self.map(delta)))


class RITS(nn.Module):
    """n_v 为补全目标（V）维度，n_in 为输入总维度（V+条件）。"""

    def __init__(self, n_v, n_in=None, hidden=64):
        super().__init__()
        n_in = n_in or n_v
        self.n_v = n_v
        self.hidden = hidden
        self.rnn = nn.GRUCell(n_in, hidden)
        self.decay = TemporalDecay(n_v, hidden)
        self.hist = nn.Linear(hidden, n_v)
        self.feat = FeatureRegression(n_v)
        self.combine = nn.Linear(n_v * 2, n_v)

    def forward(self, x, m, delta):
        b, t, _ = x.shape
        xv, mv = x[..., :self.n_v], m[..., :self.n_v]
        h = torch.zeros(b, self.hidden, device=x.device)
        c = torch.zeros(b, self.n_v, device=x.device)
        preds = []
        for i in range(t):
            gamma = self.decay(delta[:, i, :self.n_v])
            h = h * gamma
            x_hat = self.hist(h)
            v_rep = mv[:, i] * xv[:, i] + (1 - mv[:, i]) * x_hat
            cond = x[:, i, self.n_v:] * m[:, i, self.n_v:]
            z = torch.cat([v_rep, cond], dim=1)
            h = self.rnn(z, h)
            c = mv[:, i] * xv[:, i] + (1 - mv[:, i]) * c
            beta = torch.sigmoid(self.combine(torch.cat([x_hat, c], dim=1)))
            pred = beta * c + (1 - beta) * self.feat(x_hat)
            preds.append(pred)
        return torch.stack(preds, dim=1)


class BRITS(nn.Module):
    def __init__(self, n_v, n_in=None, hidden=64):
        super().__init__()
        self.fwd = RITS(n_v, n_in, hidden)
        self.bwd = RITS(n_v, n_in, hidden)

    def forward(self, x, m, delta):
        pf = self.fwd(x, m, delta)
        pb = self.bwd(
            torch.flip(x, [1]),
            torch.flip(m, [1]),
            torch.flip(delta, [1]),
        )
        return (pf + torch.flip(pb, [1])) / 2.0


# ============================ SAITS ============================

class SAITSBlock(nn.Module):
    def __init__(self, d, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(a))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class SAITS(nn.Module):
    """n_v 为补全目标（V）维度，n_in 为输入总维度（V+条件）。"""

    def __init__(self, n_v, n_in=None, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        n_in = n_in or n_v
        self.n_v = n_v
        self.b1 = SAITSBlock(n_in, n_heads, d_ff, dropout)
        self.b2 = SAITSBlock(n_in, n_heads, d_ff, dropout)
        self.head = nn.Linear(n_in, n_v)

    def forward(self, x, m, delta=None):
        h1 = self.b1(x * m)
        z1 = m * x + (1 - m) * h1
        h2 = self.b2(z1)
        v_pred = torch.tanh(self.head(h2))
        xv, mv = x[..., :self.n_v], m[..., :self.n_v]
        return mv * xv + (1 - mv) * v_pred


# ============================ 网格 MemoryUNet ============================

class MaskAwareConv2d(nn.Module):
    """partial=True 时为偏置卷积（掩码处不计权并按有效比例归一化），否则普通卷积。"""

    def __init__(self, cin, cout, k=3, stride=1, partial=True):
        super().__init__()
        self.partial = partial
        self.k = k
        self.stride = stride
        self.pad = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, m=None):
        if not self.partial:
            y = self.conv(F.pad(x, (self.pad, self.pad, self.pad, self.pad)))
            return self.act(y), m

        xp = F.pad(x * m, (self.pad, self.pad, self.pad, self.pad))
        mp = F.pad(m, (self.pad, self.pad, self.pad, self.pad))
        y = self.conv(xp)
        valid = F.conv2d(
            mp,
            torch.ones(1, 1, self.k, self.k, device=x.device),
            stride=self.stride,
        )
        scale = (self.k * self.k) / valid.clamp(min=1.0)
        y = y * scale
        return self.act(y), (valid > 0).float()


class MemoryUNet(nn.Module):
    """输入 x [B,T,C,H,W]（C=4：V 为补全目标，CNR/SW/Skew 为条件），
    m [B,T,1,H,W]；输出 [B,T,1,H,W]（只补 V）。"""

    def __init__(self, base=16, in_channels=4, partial=True):
        super().__init__()
        self.e1 = MaskAwareConv2d(in_channels, base, 3, partial=partial)
        self.e2 = MaskAwareConv2d(base, base * 2, 3, stride=2, partial=partial)
        self.e3 = MaskAwareConv2d(base * 2, base * 4, 3, stride=2, partial=partial)
        self.fwd = ConvLSTMCell(base * 4, base * 4, 3)
        self.bwd = ConvLSTMCell(base * 4, base * 4, 3)
        self.d3 = nn.Conv2d(base * 8, base * 2, 3, padding=1)
        self.d2 = nn.Conv2d(base * 4, base, 3, padding=1)
        self.d1 = nn.Conv2d(base * 2, 1, 3, padding=1)

    def forward(self, x, m):
        # x: [B,T,C,H,W]；m: [B,T,1,H,W]
        b, t = x.shape[:2]
        s1, s2, s3 = [], [], []
        for i in range(t):
            a1, m1 = self.e1(x[:, i], m[:, i])
            a2, m2 = self.e2(a1, m1)
            a3, _m3 = self.e3(a2, m2)
            s1.append(a1)
            s2.append(a2)
            s3.append(a3)

        hh, ww = s3[0].shape[-2:]
        hf = torch.zeros(b, s3[0].shape[1], hh, ww, device=x.device)
        cf = torch.zeros_like(hf)
        fo = []
        for i in range(t):
            hf, cf = self.fwd(s3[i], (hf, cf))
            fo.append(hf)
        hb = torch.zeros_like(hf)
        cb = torch.zeros_like(cf)
        bo = [None] * t
        for i in range(t - 1, -1, -1):
            hb, cb = self.bwd(s3[i], (hb, cb))
            bo[i] = hb

        out = []
        for i in range(t):
            d = torch.cat([fo[i], bo[i]], dim=1)
            d = F.interpolate(F.relu(self.d3(d)), size=s2[i].shape[-2:])
            d = torch.cat([d, s2[i]], dim=1)
            d = F.interpolate(F.relu(self.d2(d)), size=s1[i].shape[-2:])
            d = torch.cat([d, s1[i]], dim=1)
            out.append(torch.tanh(self.d1(d)))
        return torch.stack(out, dim=1)


# ============================ 3D 残差细化 ============================

def _conv3d(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class Refine3d(nn.Module):
    """输入补全 V + 掩码（[B,2,T,H,W]），输出细化后的 V（[B,1,T,H,W]）。"""

    def __init__(self, base=8):
        super().__init__()
        self.c1 = _conv3d(2, base)
        self.c2 = _conv3d(base, base * 2)
        self.up = nn.Conv3d(base * 2, base, 3, padding=1)
        self.out = nn.Conv3d(base, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, v, m):
        e1 = self.c1(torch.cat([v, m], dim=1))
        e2 = self.c2(F.avg_pool3d(e1, (1, 2, 2)))
        d = F.interpolate(self.up(e2), size=e1.shape[2:], mode="trilinear",
                          align_corners=False)
        res = torch.tanh(self.out(d + e1[:, :1]))
        return torch.clamp(v + res, -1.0, 1.0)


# ============================ 逐帧雷达补全（CGAN / DSA-UNet） ============================

def _gn(cin, cout, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
    )


class SelfAttention2d(nn.Module):
    """SAGAN 风格自注意力。"""

    def __init__(self, c):
        super().__init__()
        hc = max(c // 8, 1)
        self.q = nn.Conv2d(c, hc, 1)
        self.k = nn.Conv2d(c, hc, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).reshape(b, -1, h * w).permute(0, 2, 1)
        k = self.k(x).reshape(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.v(x).reshape(b, -1, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.gamma * out


class DilatedBlock(nn.Module):
    """多尺度空洞卷积块。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.d1 = _gn(cin, cout, 1)
        self.d2 = _gn(cout, cout, 2)
        self.d3 = _gn(cout, cout, 4)

    def forward(self, x):
        return self.d3(self.d2(self.d1(x)))


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DSAUNet(nn.Module):
    """空洞 + 自注意力 UNet。输入 x [B,4,H,W]、m [B,1,H,W]，输出 [B,1,H,W]。"""

    def __init__(self, in_channels=4, base=24):
        super().__init__()
        self.e1 = DilatedBlock(in_channels + 1, base)
        self.e2 = DilatedBlock(base, base * 2)
        self.e3 = DilatedBlock(base * 2, base * 4)
        self.bneck = nn.Sequential(
            _gn(base * 4, base * 8, 2),
            SelfAttention2d(base * 8),
            _gn(base * 8, base * 8, 2),
        )
        self.d3 = UpBlock(base * 8 + base * 4, base * 4)
        self.d2 = UpBlock(base * 4 + base * 2, base * 2)
        self.d1 = UpBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, m):
        a1 = self.e1(torch.cat([x, m], dim=1))
        a2 = self.e2(F.avg_pool2d(a1, 2))
        a3 = self.e3(F.avg_pool2d(a2, 2))
        b = self.bneck(F.avg_pool2d(a3, 2))
        d = self.d3(b, a3)
        d = self.d2(d, a2)
        d = self.d1(d, a1)
        return torch.tanh(self.out(d))


class VGGBlock(nn.Module):
    def __init__(self, cin, cmid, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cmid, 3, padding=1),
            nn.GroupNorm(8, cmid), nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cout, 3, padding=1),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class NestedUNet(nn.Module):
    """U-Net++ 生成器（嵌套密集跳连）。输入 x [B,4,H,W]、m [B,1,H,W]，输出 [B,1,H,W]。"""

    def __init__(self, in_channels=4, out_channels=1, base=24):
        super().__init__()
        nb = [base, base * 2, base * 4, base * 8]
        self.x00 = VGGBlock(in_channels + 1, nb[0], nb[0])
        self.x10 = VGGBlock(nb[0], nb[1], nb[1])
        self.x20 = VGGBlock(nb[1], nb[2], nb[2])
        self.x30 = VGGBlock(nb[2], nb[3], nb[3])
        self.x01 = VGGBlock(nb[0] + nb[1], nb[0], nb[0])
        self.x11 = VGGBlock(nb[1] + nb[2], nb[1], nb[1])
        self.x21 = VGGBlock(nb[2] + nb[3], nb[2], nb[2])
        self.x02 = VGGBlock(nb[0] * 2 + nb[1], nb[0], nb[0])
        self.x12 = VGGBlock(nb[1] * 2 + nb[2], nb[1], nb[1])
        self.x03 = VGGBlock(nb[0] * 3 + nb[1], nb[0], nb[0])
        self.out = nn.Conv2d(nb[0], out_channels, 1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, x, m):
        x = torch.cat([x, m], dim=1)
        x00 = self.x00(x)
        x10 = self.x10(F.avg_pool2d(x00, 2))
        x20 = self.x20(F.avg_pool2d(x10, 2))
        x30 = self.x30(F.avg_pool2d(x20, 2))
        x01 = self.x01(torch.cat([x00, self._up(x10, x00)], 1))
        x11 = self.x11(torch.cat([x10, self._up(x20, x10)], 1))
        x21 = self.x21(torch.cat([x20, self._up(x30, x20)], 1))
        x02 = self.x02(torch.cat([x00, x01, self._up(x11, x00)], 1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, x10)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, x00)], 1))
        return torch.tanh(self.out(x03))


class CondPatchDisc(nn.Module):
    """条件 PatchGAN：判别 (遮挡输入 x, 目标 V)。"""

    def __init__(self, cond_channels=4, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels + 1, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.InstanceNorm2d(base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 3, 1, 1),
        )

    def forward(self, cond, target):
        return self.net(torch.cat([cond, target], dim=1))


# ============================ 下游预测器（所有方法共用） ============================

def build_predictor():
    """统一的下游 ConvLSTM 预测器：输入 4 物理通道，输出未来 V。"""
    from predictor import WindFieldPredictor
    from common import cfg
    return WindFieldPredictor(
        cfg.PHYS_CHANNELS, cfg.HIDDEN_CHANNELS, 2, 3, cfg.PRED_STEPS)
