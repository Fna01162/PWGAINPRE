# -*- coding: utf-8 -*-
"""ConvLSTM Encoder-Decoder 风场预测网络。

输入：[B, Tin, Cin, H, W]  历史多通道 PPI 序列
输出：[B, Tout, 1, H, W]   未来 V 风速场
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """标准 ConvLSTM 单元，用于二维空间网格上的时序建模。"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x, cur_state=None):
        if cur_state is None or cur_state[0] is None or cur_state[1] is None:
            h_cur = torch.zeros(
                x.shape[0], self.hidden_dim,
                x.shape[-2], x.shape[-1],
                device=x.device, dtype=x.dtype,
            )
            c_cur = torch.zeros(
                x.shape[0], self.hidden_dim,
                x.shape[-2], x.shape[-1],
                device=x.device, dtype=x.dtype,
            )
        else:
            h_cur, c_cur = cur_state

        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device, dtype):
        h = torch.zeros(batch_size, self.hidden_dim, height, width,
                        device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, height, width,
                        device=device, dtype=dtype)
        return h, c


class WindFieldPredictor(nn.Module):
    """ConvLSTM Encoder-Decoder 风场预测网络。

    架构：
        空间编码 → ConvLSTM 编码器 → ConvLSTM 解码器 → 空间解码 → 输出

    输入：
        x  : [B, Tin, Cin, H, W]  历史 PPI 多通道序列
    输出：
        out: [B, Tout, 1, H, W]   未来 V 风速场
    """

    def __init__(
        self,
        input_channels=6,
        hidden_channels=64,
        num_layers=2,
        kernel_size=3,
        pred_steps=1,
        base_filter=32,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pred_steps = pred_steps

        # 空间编码：压缩通道，保持空间尺寸
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_filter, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, base_filter), base_filter),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_filter, base_filter, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, base_filter), base_filter),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ConvLSTM 编码器层
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = base_filter if i == 0 else hidden_channels
            self.encoder_cells.append(
                ConvLSTMCell(in_dim, hidden_channels, kernel_size)
            )

        # ConvLSTM 解码器层
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_cells.append(
                ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
            )

        # 空间解码：1×1 卷积输出 V
        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, base_filter, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, base_filter), base_filter),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_filter, base_filter, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, base_filter), base_filter),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_filter, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        B, Tin, Cin, H, W = x.shape

        # 空间编码：逐帧处理
        encoded_frames = []
        for t in range(Tin):
            frame = x[:, t]
            encoded_frames.append(self.spatial_encoder(frame))
        encoded = torch.stack(encoded_frames, dim=1)

        # ConvLSTM 编码
        h_t = [None] * self.num_layers
        c_t = [None] * self.num_layers
        for t in range(Tin):
            inp = encoded[:, t]
            for layer_idx in range(self.num_layers):
                h_t[layer_idx], c_t[layer_idx] = self.encoder_cells[layer_idx](
                    inp, (h_t[layer_idx], c_t[layer_idx])
                )
                inp = h_t[layer_idx]

        # ConvLSTM 解码：逐步预测未来帧
        outputs = []
        for step in range(self.pred_steps):
            inp = h_t[-1]
            for layer_idx in range(self.num_layers):
                h_t[layer_idx], c_t[layer_idx] = self.decoder_cells[layer_idx](
                    inp, (h_t[layer_idx], c_t[layer_idx])
                )
                inp = h_t[layer_idx]
            frame_out = self.spatial_decoder(h_t[-1])
            outputs.append(frame_out.unsqueeze(1))

        out = torch.cat(outputs, dim=1)
        return out


def init_predictor_weights(model, mean=0.0, std=0.02):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d,)):
            nn.init.normal_(module.weight, mean, std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)