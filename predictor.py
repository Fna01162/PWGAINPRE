import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularWConv2d(nn.Module):
    """
    [Fix M4]: 方位向 (W 维度, 0-360度) 周期填充的 2D 卷积。
    雷达 PPI 数据在方位角是首尾相接的闭环，使用循环填充(circular)可以消除物理上的断层硬边界。
    距离库 (H 维度) 没有这种周期性，因此依然使用普通的常数零填充(constant)。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CircularWConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.pad_h = kernel_size[0] // 2
        self.pad_w = kernel_size[1] // 2

        # 将原有的 Conv2d padding 设为 0，我们利用 F.pad 显式进行混合 padding
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        # F.pad 参数顺序: (W_left, W_right, H_top, H_bottom)

        # 1. 对 W 维 (方位向左右) 进行 circular 填充 (0度与360度接壤)
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode='circular')

        # 2. 对 H 维 (距离库上下) 进行 zero 填充 (雷达中心与最远探测距离为截断区域)
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode='constant', value=0.0)

        return self.conv(x)


class ConvLSTMCell(nn.Module):
    """
    使用方位向周期卷积的 ConvLSTM 单元。
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # [应用 M4]: 使用改进后的 CircularWConv2d 替代 nn.Conv2d
        self.conv = CircularWConv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=kernel_size,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # 拼接输入与当前隐藏状态
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class WindFieldPredictor(nn.Module):
    """
    雷达风场预测网络 (基于多层 ConvLSTM，集成物理先验)。
    """

    def __init__(self, input_channels, hidden_channels, num_layers, kernel_size, pred_steps):
        super(WindFieldPredictor, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pred_steps = pred_steps

        cells = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_channels if i == 0 else self.hidden_channels
            cells.append(ConvLSTMCell(input_dim=cur_input_dim,
                                      hidden_dim=self.hidden_channels,
                                      kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

        # 最终的回归映射层同样使用 CircularWConv2d
        self.final_conv = CircularWConv2d(
            in_channels=self.hidden_channels,
            out_channels=1,  # 预测目标通道数，只预测 V (径向速度)
            kernel_size=1
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 历史输入张量, Shape [B, Tin, C, H, W]
        Returns:
            predictions: 预测输出张量, Shape [B, Tout, 1, H, W]
        """
        B, Tin, C, H, W = x.shape
        device = x.device

        # 初始化 Encoder 状态
        h_t, c_t = [], []
        for i in range(self.num_layers):
            h, c = self.cells[i].init_hidden(B, (H, W), device)
            h_t.append(h)
            c_t.append(c)

        # ==========================================
        # 1. Encoder: 提取时空历史特征
        # ==========================================
        for t in range(Tin):
            in_tensor = x[:, t, :, :, :]
            for i, cell in enumerate(self.cells):
                h_t[i], c_t[i] = cell(in_tensor, (h_t[i], c_t[i]))
                in_tensor = h_t[i]

        # [Fix L2]: 深拷贝分离 Decoder 和 Encoder 的状态列表
        # 如果直接使用 h_t，多步预测时会对历史记忆造成不可逆的覆盖污染
        dec_h_t = [h.clone() for h in h_t]
        dec_c_t = [c.clone() for c in c_t]

        # ==========================================
        # 2. Decoder: 自回归预测未来序列
        # ==========================================
        predictions = []
        # 取最后一层的隐藏状态映射出第一个预测值
        dec_input = self.final_conv(dec_h_t[-1])

        # [Fix L1]: Tanh 层与 minmax 归一化[-1,1]严格对应，防止数值越界
        dec_input = torch.tanh(dec_input)

        # 用于对齐通道数 (预测目标只有 V，但网络输入有 C 个通道，缺少的用零填充)
        padding_channels = self.input_channels - 1

        for t in range(self.pred_steps):
            if padding_channels > 0:
                pad_tensor = torch.zeros(B, padding_channels, H, W, device=device)
                cur_input = torch.cat([dec_input, pad_tensor], dim=1)
            else:
                cur_input = dec_input

            for i, cell in enumerate(self.cells):
                dec_h_t[i], dec_c_t[i] = cell(cur_input, (dec_h_t[i], dec_c_t[i]))
                cur_input = dec_h_t[i]

            dec_output = self.final_conv(dec_h_t[-1])
            dec_output = torch.tanh(dec_output)

            predictions.append(dec_output)
            dec_input = dec_output  # 将当前输出作为下一步的自回归输入

        # 组装预测结果, 输出形状: [B, Tout, 1, H, W]
        predictions = torch.stack(predictions, dim=1)

        return predictions