import numpy
import torch
from config import Config
from extension_model import build_extension_model

cfg = Config()
cfg.ext_model_type = "spatiotemporal"
m = build_extension_model(cfg)

x_cur = torch.randn(2, 6, 68, 125)
x_seq = torch.randn(2, cfg.temporal_in_steps, 6, 68, 125)
y = m(x_cur, x_seq)
assert y.shape == (2, 1, 10, 125), y.shape

for mod in m.gru_cell.modules():
    if isinstance(mod, torch.nn.Conv2d):
        from untils.circular_pad import CircularWConv2d
        assert isinstance(mod, CircularWConv2d) or any(
            isinstance(parent, CircularWConv2d) for parent in m.gru_cell.modules()
        ), f"Found plain Conv2d: {type(mod).__name__}"

has_circular = any(isinstance(mod, type(m.gru_cell.conv_gates)) for mod in m.gru_cell.modules())
assert has_circular, "No CircularWConv2d found in GRU cell"

print("ok", sum(p.numel() for p in m.parameters()) / 1e6, "M params")