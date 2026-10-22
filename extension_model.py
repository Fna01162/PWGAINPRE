# -*- coding: utf-8 -*-
"""径向量程扩展网络：CNN 基线 + 改进 PPI-NAF + 时空扩展网络（修复与优化版）

任务：从近距离 PPI 风场预测远距离径向风速 V。
输入：[B, C_in, H_near, W]  近距离多通道数据
输出：[B, 1, H_far, W]     远距离 V 预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from untils.circular_pad import CircularWConv2d


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_gates = CircularWConv2d(
            input_dim + hidden_dim, 2 * hidden_dim,
            kernel_size=kernel_size, padding=1
        )
        self.conv_can = CircularWConv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=1
        )

    def forward(self, x, h_cur):
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv_gates(combined)
        z, r = torch.split(gates, self.hidden_dim, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        can_in = torch.cat([x, r * h_cur], dim=1)
        n = torch.tanh(self.conv_can(can_in))
        h_next = (1 - z) * h_cur + z * n
        return h_next


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)]
        if use_norm:
            layers.append(nn.GroupNorm(min(8, out_ch), out_ch))
        layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNRangeExtension(nn.Module):
    """CNN 基线径向量程扩展网络[cite: 1]。"""

    def __init__(self, in_channels=6, base_filter=32, far_gates=10,
                 boundary_gates=8):
        super().__init__()
        self.far_gates = far_gates
        self.boundary_gates = boundary_gates
        bf = base_filter

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, bf),
            ConvBlock(bf, bf * 2),
            ConvBlock(bf * 2, bf * 2),
        )

        self.boundary_enc = nn.Sequential(
            nn.Conv2d(bf * 2, bf * 2,
                      kernel_size=(boundary_gates, 1), padding=0),
            nn.GroupNorm(min(8, bf * 2), bf * 2),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            ConvBlock(bf * 4, bf * 2),
            ConvBlock(bf * 2, bf),
            nn.Conv2d(bf, 1, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.encoder(x)

        bd = feat[:, :, -self.boundary_gates:]
        bd = self.boundary_enc(bd)
        bd = bd.expand(-1, -1, self.far_gates, -1)

        feat_far = F.interpolate(
            feat, size=(self.far_gates, W), mode='bilinear', align_corners=False
        )

        combined = torch.cat([feat_far, bd], dim=1)
        return self.decoder(combined)


class PPINAFRangeExtension(nn.Module):
    """改进 PPI-NAF 径向量程扩展网络[cite: 1]。"""

    def __init__(self, in_channels=6, base_filter=32, far_gates=10,
                 boundary_gates=8, n_naf_blocks=4):
        super().__init__()
        self.far_gates = far_gates
        self.boundary_gates = boundary_gates
        bf = base_filter

        self.input_proj = ConvBlock(in_channels, bf)

        self.encoder_down1 = nn.Sequential(
            ConvBlock(bf, bf * 2, kernel_size=3),
            nn.Conv2d(bf * 2, bf * 2, kernel_size=(2, 1), stride=(2, 1)),
        )
        self.encoder_down2 = nn.Sequential(
            ConvBlock(bf * 2, bf * 4, kernel_size=3),
            nn.Conv2d(bf * 4, bf * 4, kernel_size=(2, 1), stride=(2, 1)),
        )

        self.naf_blocks = nn.Sequential(*[
            _NAFBlock2d(bf * 4) for _ in range(n_naf_blocks)
        ])

        self.boundary_enc = nn.Sequential(
            nn.Conv2d(bf, bf * 2,
                      kernel_size=(boundary_gates, 1), padding=0),
            nn.GroupNorm(min(8, bf * 2), bf * 2),
            nn.GELU(),
        )

        self.decoder_up1 = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, kernel_size=(2, 1), stride=(2, 1)),
            ConvBlock(bf * 2, bf * 2),
        )
        self.decoder_up2 = nn.Sequential(
            nn.ConvTranspose2d(bf * 2, bf, kernel_size=(2, 1), stride=(2, 1)),
            ConvBlock(bf, bf),
        )

        self.fusion = ConvBlock(bf * 3, bf * 2)
        self.head = nn.Sequential(
            ConvBlock(bf * 2, bf),
            nn.Conv2d(bf, 1, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        e0 = self.input_proj(x)
        e1 = self.encoder_down1(e0)
        e2 = self.encoder_down2(e1)

        e2 = self.naf_blocks(e2)

        d1 = self.decoder_up1(e2)
        if d1.shape[2] != e1.shape[2]:
            d1 = F.interpolate(d1, size=(e1.shape[2], W), mode='bilinear',
                               align_corners=False)
        d1 = d1 + e1

        d0 = self.decoder_up2(d1)
        if d0.shape[2] != e0.shape[2]:
            d0 = F.interpolate(d0, size=(e0.shape[2], W), mode='bilinear',
                               align_corners=False)
        d0 = d0 + e0

        bd = e0[:, :, -self.boundary_gates:]
        bd = self.boundary_enc(bd)
        bd = bd.expand(-1, -1, self.far_gates, -1)

        feat_far = F.interpolate(
            d0, size=(self.far_gates, W), mode='bilinear', align_corners=False
        )

        combined = self.fusion(torch.cat([feat_far, bd], dim=1))
        return self.head(combined)


class _NAFBlock2d(nn.Module):
    """简化 NAFBlock 用于 2D 特征变换[cite: 1]。"""

    def __init__(self, channels):
        super().__init__()
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        res = x
        x = self.pw1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.dw(x)
        x = self.pw2(x)
        x = self.norm(x)
        x = self.gamma * x + self.beta
        return x + res


class SpatioTemporalRangeExtension(nn.Module):
    """改进版时空径向量程扩展网络：
    1. 方位角全程 CircularWConv2d 保证 0°/360° 无缝周期连续
    2. 近-远边界多尺度趋势编码 + 远场相对距离门位置先验 (Range Pos Encoding)
    3. 边界锚点残差学习 (Anchor Residual Extrapolation)
    4. 动态尺寸支持 + 修复 Gated / Concat 双模式时空融合
    """
    def __init__(self, cfg):
        super().__init__()
        self.far_gates = cfg.far_end_gate - cfg.near_end_gate
        self.boundary_gates = min(12, cfg.near_end_gate)
        bf = getattr(cfg, "ext_base_filter", 32)
        tc = getattr(cfg, "ext_temporal_channels", 32)
        in_ch = getattr(cfg, "ext_in_channels", 6)
        if in_ch <= 0:
            in_ch = getattr(cfg, "phys_channels", 4) + 2
        self.fusion_type = getattr(cfg, "fusion_type", "gated")

        # 1. 时序分支：动态尺寸编码 + ConvGRU
        self.time_enc = nn.Sequential(
            CircularWConv2d(in_ch, tc, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, tc), tc), nn.SiLU(),
            CircularWConv2d(tc, tc, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, tc), tc), nn.SiLU(),
        )
        self.gru_cell = ConvGRUCell(tc, tc)

        # 2. 空间分支：使用方位角 Circular 卷积 + NAFBlock2d 增强特征
        self.spatial_encoder = nn.Sequential(
            CircularWConv2d(in_ch, bf, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, bf), bf), nn.GELU(),
            CircularWConv2d(bf, bf * 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, bf * 2), bf * 2), nn.GELU(),
            _NAFBlock2d(bf * 2),
            _NAFBlock2d(bf * 2),
        )

        # 3. 边界趋势编码器：聚合近场最后 boundary_gates 门的径向一阶/二阶趋势
        self.spatial_boundary = nn.Sequential(
            nn.Conv2d(bf * 2, bf * 2, kernel_size=(self.boundary_gates, 1), padding=0),
            nn.GroupNorm(min(8, bf * 2), bf * 2),
            nn.GELU(),
            CircularWConv2d(bf * 2, bf * 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, bf * 2), bf * 2),
            nn.GELU(),
        )

        # 4. 远距离相对门位置编码 (1..far_gates) 引导径向递推衰减/增长
        self.range_pos_emb = nn.Parameter(torch.randn(1, bf * 2, self.far_gates, 1) * 0.02)

        # 空间特征压缩对齐到 bf * 2
        self.spatial_proj = nn.Sequential(
            CircularWConv2d(bf * 4, bf * 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, bf * 2), bf * 2),
            nn.GELU(),
        )

        # 5. 时空融合头（支持 concat 与修复后的 gated）
        if self.fusion_type == "concat":
            self.fusion = nn.Sequential(
                CircularWConv2d(bf * 2 + tc, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64), nn.SiLU(),
                nn.Conv2d(64, 1, kernel_size=1),
            )
        elif self.fusion_type == "gated":
            self.t_proj = CircularWConv2d(tc, bf * 2, kernel_size=3, padding=1)
            self.gate = nn.Conv2d(bf * 4, bf * 2, kernel_size=1)
            self.head = nn.Sequential(
                CircularWConv2d(bf * 2, bf, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, bf), bf), nn.SiLU(),
                nn.Conv2d(bf, 1, kernel_size=1),
            )
        else:
            raise ValueError(f"未知融合类型: {self.fusion_type}")

        # 残差输出头零初始化：初始输出严格等于近场边界锚点风速，训练极度稳定
        self._zero_init_last_layer()

    def _zero_init_last_layer(self):
        last_conv = self.fusion[-1] if self.fusion_type == "concat" else self.head[-1]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, x_cur, x_seq):
        B, C, H_near, W = x_cur.shape
        T = x_seq.shape[1]

        # ---- 时序分支（动态自适应尺寸，不再硬编码 17x32） ----
        bt = x_seq.reshape(B * T, *x_seq.shape[2:])
        te = self.time_enc(bt)
        _, C_t, H_t, W_t = te.shape
        te = te.reshape(B, T, C_t, H_t, W_t)

        h = torch.zeros(B, C_t, H_t, W_t, device=x_cur.device, dtype=x_cur.dtype)
        for t in range(T):
            h = self.gru_cell(te[:, t], h)
        # 仅取时序特征的近场外侧过渡区插值到远场尺寸
        h_edge = h[:, :, max(0, H_t - 4):, :]
        F_t = F.interpolate(h_edge, size=(self.far_gates, W), mode="bilinear", align_corners=False)

        # ---- 空间分支（聚焦近-远过渡区 + 位置编码） ----
        feat_s = self.spatial_encoder(x_cur)
        bd = feat_s[:, :, -self.boundary_gates:]
        bd = self.spatial_boundary(bd)
        bd = bd.expand(-1, -1, self.far_gates, -1) + self.range_pos_emb

        # 仅用近场外圈最后 boundary_gates 门做插值外推，避免近场中心(gate 0)污染远场
        feat_edge = feat_s[:, :, -self.boundary_gates:]
        feat_far = F.interpolate(
            feat_edge, size=(self.far_gates, W), mode="bilinear", align_corners=False
        )
        F_s = self.spatial_proj(torch.cat([feat_far, bd], dim=1))  # [B, bf*2, far_gates, W]

        # ---- 边界风速锚点（取最后3个近场门的均值作为基线） ----
        v_anchor = x_cur[:, 0:1, -3:, :].mean(dim=2, keepdim=True).expand(-1, 1, self.far_gates, -1)

        # ---- 时空融合 + 残差输出 ----
        if self.fusion_type == "concat":
            delta_v = self.fusion(torch.cat([F_s, F_t], dim=1))
        else:
            tp = self.t_proj(F_t)
            g = torch.sigmoid(self.gate(torch.cat([F_s, tp], dim=1)))
            fused = g * F_s + (1.0 - g) * tp
            delta_v = self.head(fused)

        return v_anchor + delta_v


def build_extension_model(cfg):
    near_gates = cfg.near_end_gate
    far_gates = cfg.far_end_gate - cfg.near_end_gate
    phys_ch = getattr(cfg, "phys_channels", 4)
    in_ch = getattr(cfg, "ext_in_channels", 0)
    if in_ch <= 0:
        in_ch = phys_ch + 2
    bf = getattr(cfg, "ext_base_filter", 32)
    model_type = getattr(cfg, "ext_model_type", "cnn")

    if model_type == "cnn":
        model = CNNRangeExtension(
            in_channels=in_ch, base_filter=bf, far_gates=far_gates
        )
    elif model_type == "ppinaf":
        model = PPINAFRangeExtension(
            in_channels=in_ch, base_filter=bf, far_gates=far_gates
        )
    elif model_type == "spatiotemporal":
        model = SpatioTemporalRangeExtension(cfg)
    else:
        raise ValueError(f"未知扩展模型类型: {model_type}")

    if in_ch != getattr(cfg, "ext_in_channels", 0):
        cfg.ext_in_channels = in_ch
    return model


def init_extension_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)