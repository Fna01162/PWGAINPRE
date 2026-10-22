import torch
import torch.nn as nn
import torch.nn.functional as F
from untils.circular_pad import CircularWConv3d


class LayerNorm3d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / (var + self.eps).sqrt()
        return x * self.weight.view(1, -1, 1, 1, 1) + \
               self.bias.view(1, -1, 1, 1, 1)


class SimpleGate3d(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class PPIAzimuthConv3D(nn.Module):
    """
    PPI方位角循环深度卷积。

    输入：
        [B,C,T,H,W]

    只在W方向进行Circular Padding，
    H方向和T方向不扩展。

    kernel:
        1 × 1 × 5

    W对应PPI方位角，因此：
        0° <-> 360°
    """

    def __init__(self, channels, kernel_size=5, bias=True):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "PPIAzimuthConv3D的kernel_size必须为奇数"
            )

        self.padding = kernel_size // 2

        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 1, kernel_size),
            stride=1,
            padding=(0, 0, 0),
            groups=channels,
            bias=bias
        )

    def forward(self, x):
        p = self.padding

        x = F.pad(
            x,
            (
                p,
                p,
                0,
                0,
                0,
                0
            ),
            mode="circular"
        )

        return self.conv(x)


class PPIRadialConv3D(nn.Module):
    """
    PPI径向深度卷积。

    H对应径向距离方向，
    因此只在H方向扩大感受野。
    """

    def __init__(self, channels, kernel_size=5, bias=True):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "PPIRadialConv3D的kernel_size必须为奇数"
            )

        p = kernel_size // 2

        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, kernel_size, 1),
            stride=1,
            padding=(0, p, 0),
            groups=channels,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class PPI_NAFBlock3D(nn.Module):
    """
    PPI-NAFBlock3D

    在原始NAFBlock3D基础上，
    针对PPI风场的径向与方位角结构进行改造。

    核心改动：

    原始：
        3×3×3 Depthwise Conv

    PPI：
        ┌─ 1×5×1 径向分支
        │
        └─ 1×1×5 方位角Circular分支

    两个方向特征融合后，
    再进入NAF原有的SimpleGate、SCA、
    残差和FFN结构。
    """

    def __init__(
        self,
        c_in,
        c_out,
        DW_Expand=2,
        FFN_Expand=2
    ):
        super().__init__()

        if c_in != c_out:
            raise ValueError(
                "PPI_NAFBlock3D要求c_in=c_out，"
                f"当前为c_in={c_in}, c_out={c_out}"
            )

        self.c_in = c_in
        self.c_out = c_out

        dw_channel = c_in * DW_Expand

        # ==========================================
        # 第一阶段
        # ==========================================

        self.norm1 = LayerNorm3d(c_in)

        self.conv1 = nn.Conv3d(
            c_in,
            dw_channel * 2,
            1,
            1,
            0,
            bias=True
        )

        # PPI径向分支
        self.radial_conv = PPIRadialConv3D(
            dw_channel,
            kernel_size=5
        )

        # PPI方位角循环分支
        self.azimuth_conv = PPIAzimuthConv3D(
            dw_channel,
            kernel_size=5
        )

        # 径向 + 方位角融合
        self.direction_fusion = nn.Conv3d(
            dw_channel * 2,
            dw_channel,
            1,
            1,
            0,
            bias=True
        )

        # NAF SimpleGate
        self.sg = SimpleGate3d()

        # SCA
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(
                dw_channel,
                dw_channel,
                1,
                1,
                0,
                bias=True
            )
        )

        self.conv3 = nn.Conv3d(
            dw_channel,
            c_out,
            1,
            1,
            0,
            bias=True
        )

        self.beta = nn.Parameter(
            torch.zeros(
                1,
                c_out,
                1,
                1,
                1
            ),
            requires_grad=True
        )

        # ==========================================
        # NAF FFN
        # ==========================================

        ffn_channel = FFN_Expand * c_out

        self.norm2 = LayerNorm3d(c_out)

        self.conv4 = nn.Conv3d(
            c_out,
            ffn_channel * 2,
            1,
            1,
            0,
            bias=True
        )

        self.sg_ffn = SimpleGate3d()

        self.conv5 = nn.Conv3d(
            ffn_channel,
            c_out,
            1,
            1,
            0,
            bias=True
        )

        self.gamma = nn.Parameter(
            torch.zeros(
                1,
                c_out,
                1,
                1,
                1
            ),
            requires_grad=True
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.conv3.weight)

        if self.conv3.bias is not None:
            nn.init.zeros_(self.conv3.bias)

        nn.init.zeros_(self.conv5.weight)

        if self.conv5.bias is not None:
            nn.init.zeros_(self.conv5.bias)

    def forward(self, inp):

        # ==========================================
        # PPI方向特征提取
        # ==========================================

        x = self.norm1(inp)

        x = self.conv1(x)

        x_spatial, x_gate = x.chunk(
            2,
            dim=1
        )

        # 径向特征
        radial = self.radial_conv(
            x_spatial
        )

        # 方位角周期特征
        azimuth = self.azimuth_conv(
            x_spatial
        )

        # ==========================================
        # 方向特征融合
        # ==========================================

        x = torch.cat(
            [
                radial,
                azimuth
            ],
            dim=1
        )

        x = self.direction_fusion(x)

        # ==========================================
        # SimpleGate
        # ==========================================

        x = torch.cat(
            [
                x,
                x_gate
            ],
            dim=1
        )

        x = self.sg(x)

        # ==========================================
        # SCA
        # ==========================================

        x = x * self.sca(x)

        x = self.conv3(x)

        # ==========================================
        # 第一残差
        # ==========================================

        y = inp + x * self.beta

        # ==========================================
        # FFN
        # ==========================================

        x = self.norm2(y)

        x = self.conv4(x)

        x = self.sg_ffn(x)

        x = self.conv5(x)

        # ==========================================
        # 第二残差
        # ==========================================

        return y + x * self.gamma


# 兼容现有model.py
NAFBlock3D = PPI_NAFBlock3D