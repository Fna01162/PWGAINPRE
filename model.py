import torch
import torch.nn as nn
import torch.nn.functional as F
from untils.circular_pad import CircularWConv3d, CircularWConv2d


class BottleneckWithAxial(nn.Module):
    """Bottleneck：通道对齐后交替堆叠 NAFBlock3D 和 AxialWAttention3D。"""

    def __init__(self, in_dim, out_dim, num_pairs=1, use_axial=True, num_heads=4):
        super().__init__()
        if num_pairs < 1:
            raise ValueError("num_pairs 必须 >= 1")

        from untils.NAF_Block import NAFBlock3D
        from untils.axial_attn import AxialWAttention3D

        blocks = [nn.Conv3d(in_dim, out_dim, kernel_size=1)]

        for _ in range(num_pairs):
            blocks.append(NAFBlock3D(out_dim, out_dim))
            if use_axial:
                blocks.append(
                    AxialWAttention3D(
                        out_dim,
                        num_heads=num_heads
                    )
                )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class UNetGenerator(nn.Module):
    """
    条件式 3D U-Net。

    输入：
        x_bar : [B,4,T,H,W]
                V、CNR、SW、Skew 的上下文融合结果
        m     : [B,4,T,H,W]
                4 个物理量对应的有效观测掩膜

    实际输入：
        [B,8,T,H,W]

    输出：
        [B,1,T,H,W]
        仅生成风速 V。

    方位角 W 方向使用 Circular Padding，
    距离 H 方向不做周期处理。
    """

    def __init__(
        self,
        in_channels=8,
        out_channels=1,
        base_filter=48,
        use_cue=False,
        use_nafblock=True,
        use_volformer=False,
        use_axial_w=False,
        axial_num_pairs=1,
        axial_num_heads=4,
    ):
        super().__init__()

        bf = int(base_filter)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filter = bf
        self.use_cue = bool(use_cue)
        self.use_nafblock = bool(use_nafblock)
        self.use_volformer = bool(use_volformer)
        self.use_axial_w = bool(use_axial_w)

        def make_block(cin, cout, use_naf):
            if use_naf:
                from untils.NAF_Block import NAFBlock3D

                if cin != cout:
                    return nn.Sequential(
                        nn.Conv3d(cin, cout, kernel_size=1),
                        NAFBlock3D(cout, cout),
                    )

                return NAFBlock3D(cin, cout)

            return nn.Sequential(
                CircularWConv3d(cin, cout, 3),
                nn.GroupNorm(8, cout),
                nn.LeakyReLU(0.1, inplace=True),
                CircularWConv3d(cout, cout, 3),
                nn.GroupNorm(8, cout),
                nn.LeakyReLU(0.1, inplace=True),
            )

        # Encoder
        self.enc1 = make_block(
            in_channels,
            bf,
            use_naf=False
        )

        self.enc2 = make_block(
            bf,
            bf * 2,
            use_naf=use_nafblock
        )

        self.enc3 = make_block(
            bf * 2,
            bf * 4,
            use_naf=use_nafblock
        )

        # Bottleneck
        if use_axial_w:
            self.bottleneck = BottleneckWithAxial(
                bf * 4,
                bf * 8,
                num_pairs=axial_num_pairs,
                use_axial=True,
                num_heads=axial_num_heads,
            )
        else:
            self.bottleneck = make_block(
                bf * 4,
                bf * 8,
                use_naf=use_nafblock
            )

        # Optional VolFormer
        if use_volformer:
            from untils.volformer import VolSelfAttention3D
            self.vol_attn = VolSelfAttention3D(
                bf * 8,
                num_heads=4
            )
        else:
            self.vol_attn = None

        # Decoder
        self.up3 = CircularWConv3d(
            bf * 8,
            bf * 4,
            3
        )

        self.dec3 = make_block(
            bf * 8,
            bf * 4,
            use_naf=use_nafblock
        )

        self.up2 = CircularWConv3d(
            bf * 4,
            bf * 2,
            3
        )

        self.dec2 = make_block(
            bf * 4,
            bf * 2,
            use_naf=use_nafblock
        )

        self.up1 = CircularWConv3d(
            bf * 2,
            bf,
            3
        )

        self.dec1 = nn.Sequential(
            CircularWConv3d(
                bf * 2,
                bf,
                3
            ),
            nn.GroupNorm(8, bf),
            nn.LeakyReLU(0.1, inplace=True),
            CircularWConv3d(
                bf,
                out_channels,
                3
            ),
        )

    @staticmethod
    def _pool(x):
        kt = 2 if x.shape[2] >= 2 else 1

        return F.avg_pool3d(
            x,
            kernel_size=(kt, 2, 2),
            stride=(kt, 2, 2)
        )

    @staticmethod
    def _resize_like(x, ref):
        if x.shape[2:] == ref.shape[2:]:
            return x

        return F.interpolate(
            x,
            size=ref.shape[2:],
            mode="trilinear",
            align_corners=False
        )

    def forward(self, x_bar, m):
        if x_bar.ndim != 5:
            raise ValueError(
                f"x_bar 必须是 [B,C,T,H,W]，当前为 {x_bar.shape}"
            )

        if m.ndim != 5:
            raise ValueError(
                f"m 必须是 [B,C,T,H,W]，当前为 {m.shape}"
            )

        if x_bar.shape[0] != m.shape[0]:
            raise ValueError("x_bar 与 m 的 batch size 不一致")

        if x_bar.shape[2:] != m.shape[2:]:
            raise ValueError(
                f"x_bar 与 m 的空间/时间尺寸不一致："
                f"{x_bar.shape} vs {m.shape}"
            )

        if x_bar.shape[1] + m.shape[1] != self.in_channels:
            raise ValueError(
                f"Generator 输入通道数错误："
                f"x_bar={x_bar.shape[1]}，"
                f"m={m.shape[1]}，"
                f"总计={x_bar.shape[1] + m.shape[1]}，"
                f"期望={self.in_channels}"
            )

        inp = torch.cat([x_bar, m], dim=1)

        # Encoder
        e1 = self.enc1(inp)

        e2 = self.enc2(
            self._pool(e1)
        )

        e3 = self.enc3(e2)

        # 不再对 e3 做一次空间下采样。
        # 对当前 96×128 输入而言，此时空间尺寸为 24×32，
        # 可以避免过度压缩小范围缺失结构。
        b = self.bottleneck(e3)

        if self.vol_attn is not None:
            b = b + self.vol_attn(b)

        # Decoder
        d3 = self.up3(b)
        d3 = self._resize_like(d3, e3)
        d3 = self.dec3(
            torch.cat([d3, e3], dim=1)
        )

        d2 = self.up2(d3)
        d2 = self._resize_like(d2, e2)
        d2 = self.dec2(
            torch.cat([d2, e2], dim=1)
        )

        d1 = self.up1(d2)
        d1 = self._resize_like(d1, e1)
        d1 = self.dec1(
            torch.cat([d1, e1], dim=1)
        )

        return torch.tanh(d1)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN 判别器。

    输入：
        v     : 风速 V
        hint  : GAIN Hint

    拼接后：
        [B,2,H,W]

    输出：
        Patch logits。
    """

    def __init__(self, in_channels=2, base_filter=24):
        super().__init__()

        bf = int(base_filter)

        self.net = nn.Sequential(
            CircularWConv2d(
                in_channels,
                bf,
                4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            CircularWConv2d(
                bf,
                bf * 2,
                4,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(
                8,
                bf * 2
            ),
            nn.LeakyReLU(0.2, inplace=True),

            CircularWConv2d(
                bf * 2,
                bf * 4,
                4,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(
                8,
                bf * 4
            ),
            nn.LeakyReLU(0.2, inplace=True),

            CircularWConv2d(
                bf * 4,
                bf * 4,
                3,
                stride=1,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            CircularWConv2d(
                bf * 4,
                1,
                3,
                stride=1,
                padding=1
            ),
        )

    def forward(self, v, hint):
        if v.ndim != 4:
            raise ValueError(
                f"v 必须是 [B,1,H,W]，当前为 {v.shape}"
            )

        if hint.ndim != 4:
            raise ValueError(
                f"hint 必须是 [B,1,H,W]，当前为 {hint.shape}"
            )

        if v.shape[0] != hint.shape[0]:
            raise ValueError("v 与 hint 的 batch size 不一致")

        if v.shape[2:] != hint.shape[2:]:
            raise ValueError(
                f"v 与 hint 尺寸不一致："
                f"{v.shape} vs {hint.shape}"
            )

        return self.net(
            torch.cat([v, hint], dim=1)
        )


def init_weights(net, mean=0.0, std=0.02):
    for module in net.modules():
        if isinstance(
            module,
            (nn.Conv2d, nn.Conv3d)
        ):
            nn.init.normal_(
                module.weight,
                mean,
                std
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0
                )

        elif isinstance(
            module,
            (
                nn.GroupNorm,
                nn.BatchNorm2d,
                nn.BatchNorm3d
            )
        ):
            if module.weight is not None:
                nn.init.ones_(
                    module.weight
                )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )