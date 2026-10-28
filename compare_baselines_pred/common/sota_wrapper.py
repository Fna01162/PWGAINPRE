# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class SOTABaselineWrapper(nn.Module):
    """
    通用基线模型适配器 (Universal Adapter for OpenSTL Baselines)
    解决输入通道、空间对齐 (Circular Padding 等) 和输出截取问题。
    """
    def __init__(self, core_model, in_channels=6, target_h=96, target_w=128, ori_h=85, ori_w=125):
        super(SOTABaselineWrapper, self).__init__()
        self.core_model = core_model
        
        self.target_h = target_h
        self.target_w = target_w
        self.ori_h = ori_h
        self.ori_w = ori_w
        
        self.pad_h = target_h - ori_h  # 96 - 85 = 11
        self.pad_w = target_w - ori_w  # 128 - 125 = 3

    def forward(self, hx, hm, hr, art_mask, cfg):
        # 1. 物理隔离与通道组装 (切断真值泄露)
        valid_mask = hm[:, 0:1, :, :, :] * hr[:, 0:1, :, :, :] * art_mask
        v_channel = hx[:, 0:1, :, :, :] * valid_mask
        features = hx[:, 1:cfg.phys_channels, :, :, :]
        
        # 组装 6 通道输入: [B, 6, Tin, H, W]
        pred_in = torch.cat([v_channel, features, hm[:, 0:1], hr[:, 0:1]], dim=1)
        
        # 转换为 OpenSTL 要求的时间优先形状: [B, Tin, Channels, H, W]
        pred_in = pred_in.permute(0, 2, 1, 3, 4).contiguous() 
        
        # 2. 空间对齐 (使用循环填充解决方位角 0-360度 断层)
        # F.pad 参数顺序: (W_left, W_right, H_top, H_bottom)
        # 对方位向 W 使用 circular 填充，对距离向 H 使用 constant 填充
        if self.pad_w > 0:
            pred_in = F.pad(pred_in, (0, self.pad_w, 0, 0), mode='circular')
        if self.pad_h > 0:
            pred_in = F.pad(pred_in, (0, 0, 0, self.pad_h), mode='constant', value=0.0)
            
        # 3. 运行核心模型
        out_pred = self.core_model(pred_in)
        if isinstance(out_pred, tuple):
            out_pred = out_pred[0]  # 处理返回 hidden_states 的情况
            
        # 4. 裁剪回原尺寸与目标通道剥离
        out_pred = out_pred[:, :, :, :self.ori_h, :self.ori_w]
        
        # 最终输出目标 V (第 0 个通道): [B, Tout, 1, H, W]
        final_v_pred = out_pred[:, :, 0:1, :, :]
        
        return final_v_pred


def get_sota_model(model_name, cfg):
    """
    通过 OpenSTL 统一加载模型
    所有模型输入已通过 Wrapper 对齐为 6 通道，空间 96x128。
    """
    in_shape = (cfg.input_steps, 6, 96, 128)
    
    try:
        if model_name == "SimVP":
            from openstl.models import SimVP_Model
            core_model = SimVP_Model(in_shape=in_shape, hid_S=64, hid_T=256, N_T=4, N_S=2)
        elif model_name == "Earthformer":
            from openstl.models import EarthFormer_Model
            core_model = EarthFormer_Model(in_shape=in_shape)
        elif model_name == "PhyDNet":
            from openstl.models import PhyDNet_Model
            core_model = PhyDNet_Model(in_shape=in_shape, num_hidden=[64, 64, 64])
        elif model_name == "PredRNN":
            from openstl.models import PredRNN_Model
            core_model = PredRNN_Model(in_shape=in_shape, num_hidden=[64, 64, 64], filter_size=3, stride=1)
        elif model_name == "SwinLSTM":
            from openstl.models import SwinLSTM_Model
            core_model = SwinLSTM_Model(in_shape=in_shape, patch_size=2, window_size=4)
        else:
            raise ValueError(f"未知的基线模型: {model_name}")
            
    except ImportError:
        raise ImportError("请确保环境中已安装 OpenSTL: pip install openstl")
        
    return SOTABaselineWrapper(core_model, in_channels=6)