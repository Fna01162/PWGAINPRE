# -*- coding: utf-8 -*-
"""ConvLSTM 时间模式可视化工具（精简版）

用法:
    python viz_convlstm.py              # 生成所有可视化
    python viz_convlstm.py --sample 1   # 可视化第2个样本
"""
import os, sys, numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class SimpleVisualizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from config import Config
        from train_joint import SlidingWindowDataset
        from untils.data import Normalizer, make_obs_rel
        from predictor import WindFieldPredictor

        self.cfg = Config()
        norm = Normalizer(self.cfg.normalize, self.cfg.clip_percentile, self.cfg.phys_channels)
        arr = np.load(self.cfg.train_npy)
        obs, rel = make_obs_rel(arr, self.cfg.phys_channels)
        self.norm = norm.fit(arr, obs)

        self.ds = SlidingWindowDataset(self.cfg.val_npy, self.norm,
                                      self.cfg.input_steps, 1)
        self.loader = torch.utils.data.DataLoader(self.ds, batch_size=1, shuffle=False)

        # 初始化模型（随机权重用于演示）
        # 根据实际数据调整输入通道数
        sample_hx, *_ = next(iter(self.loader))
        actual_channels = sample_hx.shape[-1] if sample_hx.shape[-1] < sample_hx.shape[2] else sample_hx.shape[2]
        total_input_channels = actual_channels + 2  # data channels + mask + reliability
        print(f'  Model: input_channels={total_input_channels}')
        self.P = WindFieldPredictor(input_channels=total_input_channels, hidden_channels=64,
                                   num_layers=2, pred_steps=1).to(self.device)

        # 存储中间状态
        self.states = {'H': [], 'C': [], 'f': [], 'i': [], 'o': []}

    def _hook_fn(self, name):
        def hook(module, inp, outp):
            x, (h, c) = inp
            h_out, c_out = outp

            # 跳过初始化步骤（当 h 或 c 为 None 时）
            if h is None or c is None:
                return

            combined = torch.cat([x, h], dim=1)
            gates = module.conv(combined)
            gi, gf, go, gg = torch.split(gates, module.hidden_dim, dim=1)

            self.states['H'].append((name, h_out.detach().cpu()))
            self.states['C'].append((name, c_out.detach().cpu()))
            self.states['f'].append((name, torch.sigmoid(gf).detach().cpu()))
            self.states['i'].append((name, torch.sigmoid(gi).detach().cpu()))
            self.states['o'].append((name, torch.sigmoid(go).detach().cpu()))
        return hook

    def run(self, sample_idx=0):
        for idx, batch in enumerate(self.loader):
            if idx != sample_idx:
                continue

            hx, fx, hm, hr, fm, fr = [v.to(self.device) for v in batch]
            print(f'  Original shapes: hx={hx.shape}, hm={hm.shape}')

            # 转换维度到 [B,T,C,H,W]
            if hx.ndim == 5 and hx.shape[-1] <= hx.shape[2]:  # [B,T,H,W,C] -> [B,T,C,H,W]
                hx = hx.permute(0, 1, 4, 2, 3)
                fx = fx.permute(0, 1, 4, 2, 3)

            print(f'  After permute: hx={hx.shape}, hm={hm.shape}')

            # 注册钩子
            for i, cell in enumerate(self.P.encoder_cells):
                cell.register_forward_hook(self._hook_fn(f'enc_{i}'))
            for i, cell in enumerate(self.P.decoder_cells):
                cell.register_forward_hook(self._hook_fn(f'dec_{i}'))

            with torch.no_grad():
                inp = torch.cat([hx, hm, hr], dim=2)
                pred = self.P(inp)

            return {
                'history': hx.cpu(), 'future': fx.cpu(),
                'pred': pred.cpu(), 'mask': hm.cpu(),
                'states': self.states
            }
        return None

    def visualize_all(self, sample_idx=0, save_dir='vis_outputs'):
        os.makedirs(save_dir, exist_ok=True)
        results = self.run(sample_idx)
        if not results:
            print('Sample not found!')
            return

        print(f'\n=== Generating Visualizations for Sample {sample_idx} ===\n')

        # 1. 隐藏状态演化
        print('[1/4] Hidden State Evolution...')
        H_states = results['states']['H']
        n = min(len(H_states), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('ConvLSTM Hidden State H_t Evolution', fontsize=14)
        for i in range(n):
            r, c = i // 3, i % 3
            name, h = H_states[i]
            im = axes[r,c].imshow(h[0,0].numpy(), cmap='viridis')
            axes[r,c].set_title(f'{name}: mean={h[0,0].mean():.3f}')
            axes[r,c].axis('off')
        plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/hidden_states_s{sample_idx}.png', dpi=150)
        plt.close()
        print(f'  ✓ Saved: {save_dir}/hidden_states_s{sample_idx}.png')

        # 2. 门控机制
        print('[2/4] Gate Mechanisms...')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Gate Mechanisms (Last Encoder Layer)', fontsize=14)
        for idx, (gate_name, color) in enumerate(zip(['f','i','o'], ['Reds','Greens','Blues'])):
            g = results['states'][gate_name][-1][1]
            im = axes[idx].imshow(g[0].mean(0).numpy(), cmap=color, vmin=0, vmax=1)
            axes[idx].set_title(['Forget Gate f_t', 'Input Gate i_t', 'Output Gate o_t'][idx])
            axes[idx].axis('off')
        plt.colorbar(im, ax=axes.tolist(), shrink=0.6)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gates_s{sample_idx}.png', dpi=150)
        plt.close()
        print(f'  ✓ Saved: {save_dir}/gates_s{sample_idx}.png')

        # 3. 预测 vs 真实
        print('[3/4] Prediction vs Ground Truth...')
        pred = results['pred'][0,0,0].numpy()
        true = results['future'][0,0,0].numpy()
        mask = results['mask'][0,-1,0].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Prediction Comparison', fontsize=14)
        im0 = axes[0].imshow(true, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title('Ground Truth'); axes[0].axis('off')
        im1 = axes[1].imshow(pred, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Prediction'); axes[1].axis('off')
        error = np.abs(pred - true)
        im2 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=0.5)
        mae = error[mask>0.5].mean() if mask.sum()>0 else error.mean()
        axes[2].set_title(f'Absolute Error (MAE={mae:.4f})'); axes[2].axis('off')
        for im in [im0, im1, im2]:
            plt.colorbar(im, ax=axes[[im0,im1,im2].index(im)], shrink=0.8)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_s{sample_idx}.png', dpi=150)
        plt.close()
        print(f'  ✓ Saved: {save_dir}/prediction_s{sample_idx}.png')

        # 4. 时间序列
        print('[4/4] Temporal Sequence...')
        hist = results['history'][0].numpy()
        T = hist.shape[0]
        fig, axes = plt.subplots(2, (T+2)//2+1, figsize=(20, 8))
        fig.suptitle('Temporal Sequence: History → Prediction → Truth', fontsize=14)
        axes = axes.flatten()
        for t in range(T):
            axes[t].imshow(hist[t,0], cmap='RdBu_r', vmin=-1, vmax=1)
            axes[t].set_title(f't-{T-t}'); axes[t].axis('off')
        axes[T].imshow(pred, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[T].set_title('Prediction', fontweight='bold', color='blue'); axes[T].axis('off')
        axes[T+1].imshow(true, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[T+1].set_title('Truth', fontweight='bold', color='green'); axes[T+1].axis('off')
        for i in range(T+2, len(axes)): axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sequence_s{sample_idx}.png', dpi=150)
        plt.close()
        print(f'  ✓ Saved: {save_dir}/sequence_s{sample_idx}.png')

        print(f'\n✓ All visualizations saved to {save_dir}/')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=0, help='Sample index')
    parser.add_argument('--dir', type=str, default='vis_outputs', help='Save directory')
    args = parser.parse_args()

    viz = SimpleVisualizer()
    viz.visualize_all(args.sample, args.dir)