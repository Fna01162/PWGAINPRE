# -*- coding: utf-8 -*-
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from untils.data import Normalizer, crop_to, make_obs_rel, pad_to
from model import UNetGenerator
from config import Config
from train import build_contextual_fusion, build_generator_inputs, generate_contextual_cue
from untils.refiner import ResidualRefiner, apply_refiner
from untils.data import BAD_RAYS, PAD_LEFT

TARGET_CHANNEL = 0


def fit_normalizer_from_npy(npy_path, method, clip_percentile, phys_channels):
    print(f"  [归一化重拟合] 用 {npy_path} 重新计算归一化参数...")
    arr = np.load(npy_path)
    obs, _ = make_obs_rel(arr, phys_channels)
    norm = Normalizer(method, clip_percentile, phys_channels).fit(arr, obs)
    for c in range(phys_channels):
        print(f"    通道{c}: lo={norm.lo[c]:.4f}, hi={norm.hi[c]:.4f}")
    return norm


def impute_sequence(model, arr, normalizer, cfg, device, deterministic=True, R=None, return_gain=False, return_raw=False):
    T, H, W = arr.shape[:3]
    C = cfg.phys_channels
    Tf = cfg.time_steps
    target_t = cfg.target_idx

    obs, rel = make_obs_rel(arr, C)

    phys = np.nan_to_num(arr[..., :C].astype(np.float32), nan=0.0)
    norm = normalizer.transform(phys)
    norm = np.where(obs[..., None] == 1, norm, 0.0)

    pl, pr = target_t, Tf - 1 - target_t
    ext = np.pad(
        norm,
        ((pl, pr), (0, 0), (0, 0), (0, 0)),
        mode="reflect"
    )
    ext_obs = np.pad(
        obs,
        ((pl, pr), (0, 0), (0, 0)),
        mode="reflect"
    )
    ext_rel = np.pad(
        rel,
        ((pl, pr), (0, 0), (0, 0)),
        mode="reflect"
    )

    valid_base = np.ones((T, H, W), dtype=np.float32)
    ext_valid = np.pad(
        valid_base,
        ((pl, pr), (0, 0), (0, 0)),
        mode="reflect"
    )

    completed_gain = np.zeros((T, H, W), dtype=np.float32)
    completed_final = np.zeros((T, H, W), dtype=np.float32)

    model.eval()
    if R is not None:
        R.eval()

    use_cue = getattr(cfg, "use_cue", False)
    use_refiner = R is not None and getattr(cfg, "use_refiner", False)

    with torch.no_grad():
        for i in range(T):
            x_t = torch.from_numpy(
                pad_to(ext[i:i + Tf].transpose(3, 0, 1, 2))
            ).float().unsqueeze(0).to(device)

            m_t = torch.from_numpy(
                pad_to(ext_obs[i:i + Tf])
            ).float().unsqueeze(0).to(device)

            r_t = torch.from_numpy(
                pad_to(ext_rel[i:i + Tf])
            ).float().unsqueeze(0).to(device)

            v_t = torch.from_numpy(
                pad_to(ext_valid[i:i + Tf])
            ).float().unsqueeze(0).to(device)

            if m_t.ndim == 4:
                m_t = m_t.unsqueeze(1).repeat(1, C, 1, 1, 1)

            if deterministic:
                z = torch.zeros_like(x_t)
            else:
                z = torch.randn_like(x_t)

            effective_mask = m_t.clone()

            if use_cue:
                cue_t = generate_contextual_cue(
                    x_t,
                    effective_mask,
                    target_t,
                    target_channel=TARGET_CHANNEL
                )
            else:
                cue_t = torch.zeros(
                    x_t.shape[0],
                    1,
                    x_t.shape[3],
                    x_t.shape[4],
                    device=device
                )

            x_bar = build_contextual_fusion(
                x_t,
                effective_mask,
                cue_t,
                z,
                cfg.noise_scale
            )

            g_x, g_m = build_generator_inputs(
                x_bar,
                effective_mask
            )

            v_gain = model(g_x, g_m)[:, :, target_t]

            v_final = v_gain

            if use_refiner:
                obs_t = m_t[:, :1, target_t]
                rel_t = r_t[:, target_t:target_t + 1]
                valid_t = v_t[:, target_t:target_t + 1]

                # 坏射线保护开关（默认开启）：
                # Refiner 只在人工缺失区(有真值)做残差监督，坏射线无真值、
                # 训练 hole 从未覆盖，推理时 delta 属无监督外推会破坏 GAIN 的补全；
                # 故默认把坏射线视为已知可靠区、保持 GAIN 输出。
                if getattr(cfg, "protect_bad_rays_refiner", True) and BAD_RAYS:
                    obs_t = obs_t.clone()
                    rel_t = rel_t.clone()
                    for br in BAD_RAYS:
                        j = int(br) + PAD_LEFT
                        if 0 <= j < obs_t.shape[-1]:
                            obs_t[..., j] = 1.0
                            rel_t[..., j] = 1.0

                v_final = apply_refiner(
                    R,
                    v_gain,
                    x_t,
                    obs_t,
                    rel_t,
                    valid_t,
                    cfg,
                    device
                )

            gain_np = crop_to(
                v_gain[0, 0].cpu().numpy()
            )

            final_np = crop_to(
                v_final[0, 0].cpu().numpy()
            )

            completed_gain[i] = gain_np
            completed_final[i] = final_np

    completed_gain = normalizer.inverse_transform(
        np.stack([completed_gain], axis=-1)
    )[..., 0]

    completed_final = normalizer.inverse_transform(
        np.stack([completed_final], axis=-1)
    )[..., 0]

    out = arr.copy()

    out[..., TARGET_CHANNEL] = np.where(
        obs.astype(bool),
        arr[..., TARGET_CHANNEL],
        completed_final
    )

    if return_gain:
        gain_out = arr.copy()
        gain_out[..., TARGET_CHANNEL] = np.where(
            obs.astype(bool),
            arr[..., TARGET_CHANNEL],
            completed_gain
        )
        if return_raw:
            return out, gain_out, completed_gain, completed_final
        return out, gain_out

    if return_raw:
        return out, completed_gain, completed_final

    return out


def load_models(model_path, device):
    ckpt = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    class Cfg:
        pass

    cfg = Cfg()

    for k, v in ckpt["cfg"].items():
        setattr(cfg, k, v)

    use_cue = getattr(cfg, "use_cue", False)

    G = UNetGenerator(
        8,
        1,
        cfg.base_filter,
        use_cue,
        use_nafblock=getattr(cfg, "use_nafblock", False),
        use_volformer=getattr(cfg, "use_volformer", False),
        use_axial_w=getattr(cfg, "use_axial_w", False),
        axial_num_pairs=getattr(cfg, "axial_num_pairs", 1),
        axial_num_heads=getattr(cfg, "axial_num_heads", 4),
    ).to(device)

    G.load_state_dict(ckpt["model"])
    G.eval()

    R = None

    if getattr(cfg, "use_refiner", False) and "refiner" in ckpt:
        refiner_channels = 13 if getattr(cfg, "use_wavelet", True) else 9

        R = ResidualRefiner(
            refiner_channels,
            getattr(cfg, "refiner_base_filter", 16)
        ).to(device)

        R.load_state_dict(ckpt["refiner"])
        R.eval()

        print("  [Refiner] 已加载第二阶段残差细化器")
    elif getattr(cfg, "use_refiner", False):
        print("  [Refiner] 配置启用，但 checkpoint 中没有 refiner 权重")
        print("  [Refiner] 当前将只使用 GAIN_NAF")

    return G, R, cfg, ckpt


def impute_file(
    model_path,
    in_path,
    out_path,
    device,
    deterministic=True,
    refit_npy=None,
    save_gain=False
):
    print(f"[模型] {model_path}")
    print(f"[输入] {in_path}")

    G, R, cfg, ckpt = load_models(
        model_path,
        device
    )

    if refit_npy is not None:
        normalizer = fit_normalizer_from_npy(
            refit_npy,
            cfg.normalize,
            cfg.clip_percentile,
            cfg.phys_channels
        )
    else:
        normalizer = Normalizer(
            cfg.normalize,
            cfg.clip_percentile,
            cfg.phys_channels
        )
        norm_state = ckpt["normalizer"]
        if isinstance(norm_state, dict):
            normalizer.__dict__.update(norm_state)
        else:
            raise TypeError(
                f"checkpoint 归一化状态格式不支持: {type(norm_state)}"
            )
        print("  [归一化] 从 checkpoint 加载训练域归一化参数")

    arr = np.load(in_path)

    print(f"[输入 shape] {arr.shape}")
    print(f"[输入缺失点] {(arr[..., 7] == 0).sum()}")

    out, gain_out, raw_gain, raw_final = impute_sequence(
        G,
        arr,
        normalizer,
        cfg,
        device,
        deterministic=deterministic,
        R=R,
        return_gain=True,
        return_raw=True
    )

    np.save(out_path, out)

    print(f"[输出] {out_path}")
    print(f"[输出 shape] {out.shape}")

    if save_gain:
        gain_path = str(
            Path(out_path).with_name(
                Path(out_path).stem + "_gain.npy"
            )
        )
        np.save(gain_path, gain_out)
        print(f"[GAIN结果] {gain_path}")

        gain_raw_path = str(
            Path(out_path).with_name(
                Path(out_path).stem + "_gain_raw.npy"
            )
        )
        np.save(gain_raw_path, raw_gain)
        print(f"[GAIN原始预测(全位置)] {gain_raw_path}")

    if R is not None:
        print("[推理链路] GAIN_NAF → ResidualRefiner → V_final")
    else:
        print("[推理链路] GAIN_NAF → V_gain")


def main():
    cfg = Config()

    Path(cfg.output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    stem = Path(cfg.input_npy).stem

    out_path = (
        Path(cfg.output_dir)
        / f"{stem}_completed_epoch{cfg.epochs}_patience{cfg.patience}.npy"
    )

    refit_npy = getattr(
        cfg,
        "normalizer_refit_npy",
        None
    )

    if refit_npy:
        print(
            f"[零样本跨域模式] 归一化参数将用 {refit_npy} 重新拟合"
        )
    else:
        print(
            "[域内模式] 归一化参数从 checkpoint 加载"
        )

    impute_file(
        cfg.model_path,
        cfg.input_npy,
        str(out_path),
        cfg.device,
        deterministic=getattr(cfg, "deterministic", True),
        refit_npy=refit_npy,
        save_gain=True
    )


if __name__ == "__main__":
    main()


def load_joint_models(model_path, device):
    """加载联合训练检查点：G, P, R, cfg, ckpt。"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in ckpt["cfg"].items():
        setattr(cfg, k, v)

    use_cue = getattr(cfg, "use_cue", False)

    G = UNetGenerator(
        8, 1, cfg.base_filter, use_cue,
        use_nafblock=getattr(cfg, "use_nafblock", False),
        use_volformer=getattr(cfg, "use_volformer", False),
        use_axial_w=getattr(cfg, "use_axial_w", False),
        axial_num_pairs=getattr(cfg, "axial_num_pairs", 1),
        axial_num_heads=getattr(cfg, "axial_num_heads", 4),
    ).to(device)
    G.load_state_dict(ckpt["model"])
    G.eval()

    from predictor import WindFieldPredictor
    phys_ch = int(getattr(cfg, "phys_channels", 4))
    pred_in = int(getattr(cfg, "pred_input_channels", 0))
    if pred_in <= 0:
        pred_in = phys_ch + 2

    P = WindFieldPredictor(
        input_channels=pred_in,
        hidden_channels=int(getattr(cfg, "pred_hidden_channels", 64)),
        num_layers=int(getattr(cfg, "pred_num_layers", 2)),
        kernel_size=int(getattr(cfg, "pred_kernel_size", 3)),
        pred_steps=int(getattr(cfg, "pred_steps", 1)),
        base_filter=32,
    ).to(device)

    if "predictor" in ckpt:
        P.load_state_dict(ckpt["predictor"])
        print("  [Predictor] 已加载预测网络权重")
    else:
        print("  [Predictor] 检查点中没有预测网络权重，使用随机初始化")
    P.eval()

    R = None
    if getattr(cfg, "use_refiner", False) and "refiner" in ckpt:
        refiner_channels = 13 if getattr(cfg, "use_wavelet", True) else 9
        R = ResidualRefiner(refiner_channels, getattr(cfg, "refiner_base_filter", 16)).to(device)
        R.load_state_dict(ckpt["refiner"])
        R.eval()
        print("  [Refiner] 已加载第二阶段残差细化器")

    return G, P, R, cfg, ckpt


@torch.no_grad()
def evaluate_prediction(G, P, R, npy_path, normalizer, cfg, device):
    """在测试数据上评估联合模型的补全和预测性能。

    返回：
        dict: 包含补全和预测的 MAE/RMSE 及分步指标。
    """
    from untils.data import SlidingWindowDataset
    from train_joint import assemble_predictor_input
    from untils.losses import compute_prediction_metrics, masked_prediction_loss

    input_steps = int(getattr(cfg, "input_steps", 10))
    pred_steps = int(getattr(cfg, "pred_steps", 1))

    ds = SlidingWindowDataset(
        npy_path, normalizer=normalizer,
        input_steps=input_steps, pred_steps=pred_steps,
        phys_channels=cfg.phys_channels,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    G.eval()
    P.eval()
    if R is not None:
        R.eval()

    comp_mae_sum = comp_mse_sum = comp_cnt = 0.0
    pred_mae_sum = pred_mse_sum = pred_cnt = 0.0
    pred_step_mae = [0.0] * pred_steps
    pred_step_mse = [0.0] * pred_steps
    pred_step_cnt = [0.0] * pred_steps
    all_preds = []
    all_targets = []
    all_future_masks = []

    for batch in tqdm(loader, desc="pred_test"):
        history_x, history_m, history_r, future_x, future_m, future_r = batch
        history_x = history_x.to(device)
        history_m = history_m.to(device)
        history_r = history_r.to(device)
        future_x = future_x.to(device)
        future_m = future_m.to(device)
        future_r = future_r.to(device)

        B, C, Tin, H, W = history_x.shape
        t_target = Tin - 1

        v_obs = history_m[:, 0:1, t_target]
        v_rel = history_r[:, 0:1, t_target]
        v_valid = history_r[:, 0:1, t_target]
        ve = v_rel * v_valid

        from train import make_training_cue, build_contextual_fusion, build_generator_inputs

        z = torch.zeros_like(history_x)
        xin = history_x.clone()
        min_ = history_m.clone()

        cue, em = make_training_cue(xin, min_, history_r, history_r, t_target)
        if not getattr(cfg, "use_cue", False):
            cue = torch.zeros(B, 1, H, W, device=device, dtype=xin.dtype)

        xb = build_contextual_fusion(xin, em, cue, z, 0.0)
        gx, gm = build_generator_inputs(xb, em)

        vf_all = G(gx, gm)

        v_completed = torch.zeros_like(vf_all[:, 0:1])
        for t in range(Tin):
            vm = history_m[:, 0:1, t]
            v_obs_t = history_x[:, 0:1, t]
            v_gen_t = vf_all[:, 0:1, t]
            v_completed[:, :, t] = vm * v_obs_t + (1 - vm) * v_gen_t

        pred_input = assemble_predictor_input(
            history_x, v_completed, history_m, history_r
        )

        future_pred = P(pred_input)

        future_v = future_x[:, 0:1]
        future_valid_mask = future_m[:, 0:1] * future_r[:, 0:1]

        all_preds.append(future_pred.cpu().numpy())
        all_targets.append(future_v.cpu().numpy())
        all_future_masks.append(future_valid_mask.cpu().numpy())

        pred_mae_val, pred_rmse_val, pred_pts = compute_prediction_metrics(
            future_pred, future_v, future_valid_mask
        )
        pred_mae_sum += pred_mae_val * pred_pts
        pred_mse_sum += (pred_rmse_val ** 2) * pred_pts
        pred_cnt += pred_pts

        for k in range(pred_steps):
            fp = future_pred[:, k:k + 1]
            fv = future_v[:, k:k + 1]
            fm = future_valid_mask[:, k:k + 1]
            sm, sr, sp = compute_prediction_metrics(fp, fv, fm)
            pred_step_mae[k] += sm * sp
            pred_step_mse[k] += (sr ** 2) * sp
            pred_step_cnt[k] += sp

        if R is not None and getattr(cfg, "use_refiner", False):
            pass

    pred_mae = pred_mae_sum / max(pred_cnt, 1.0)
    pred_rmse = np.sqrt(pred_mse_sum / max(pred_cnt, 1.0))

    step_results = {}
    for k in range(pred_steps):
        if pred_step_cnt[k] > 0:
            step_results[f"step_{k}_MAE"] = pred_step_mae[k] / pred_step_cnt[k]
            step_results[f"step_{k}_RMSE"] = np.sqrt(pred_step_mse[k] / pred_step_cnt[k])
            step_results[f"step_{k}_points"] = pred_step_cnt[k]
        else:
            step_results[f"step_{k}_MAE"] = float("nan")
            step_results[f"step_{k}_RMSE"] = float("nan")
            step_results[f"step_{k}_points"] = 0

    return {
        "PRED_MAE": pred_mae,
        "PRED_RMSE": pred_rmse,
        "pred_valid_points": pred_cnt,
        **step_results,
        "all_preds": np.concatenate(all_preds, axis=0) if all_preds else None,
        "all_targets": np.concatenate(all_targets, axis=0) if all_targets else None,
        "all_future_masks": np.concatenate(all_future_masks, axis=0) if all_future_masks else None,
    }


def pred_test_main():
    """联合模型预测推理入口。

    用法：python main.py pred_test

    在 config.py 中配置：
        - self.model_path: 联合训练检查点路径
        - self.input_npy:  测试数据路径
        - self.output_dir: 输出目录
    """
    from train import fit_normalizer

    cfg = Config()
    device = torch.device(cfg.device)

    model_path = getattr(cfg, "model_path", "")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            f"联合模型检查点不存在: {model_path}\n"
            f"请在 config.py 中设置 self.model_path 指向联合训练的 .pth 文件"
        )

    print(f"[模型] {model_path}")
    print(f"[输入] {cfg.input_npy}")

    G, P, R, cfg_loaded, ckpt = load_joint_models(model_path, device)

    normalizer = Normalizer(
        cfg_loaded.normalize, cfg_loaded.clip_percentile, cfg_loaded.phys_channels
    )
    norm_state = ckpt["normalizer"]
    if isinstance(norm_state, dict):
        normalizer.__dict__.update(norm_state)
    print("  [归一化] 从 checkpoint 加载归一化参数")

    input_steps = int(getattr(cfg_loaded, "input_steps", 10))
    pred_steps = int(getattr(cfg_loaded, "pred_steps", 1))
    print(f"  历史帧数={input_steps}, 预测帧数={pred_steps}")

    result = evaluate_prediction(
        G, P, R, cfg.input_npy, normalizer, cfg_loaded, device
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("联合模型评估结果")
    print(f"{'='*60}")
    print(f"总体预测 MAE  = {result['PRED_MAE']:.6f}")
    print(f"总体预测 RMSE = {result['PRED_RMSE']:.6f}")
    print(f"有效标签点数   = {result['pred_valid_points']}")
    print(f"\n分步预测指标：")
    for k in range(pred_steps):
        print(f"  第{k+1}步: MAE={result[f'step_{k}_MAE']:.6f}, "
              f"RMSE={result[f'step_{k}_RMSE']:.6f}, "
              f"点数={result[f'step_{k}_points']}")

    metrics_path = out_dir / "prediction_metrics.npy"
    metrics_out = {k: v for k, v in result.items()
                   if not k.startswith("all_")}
    np.save(metrics_path, metrics_out, allow_pickle=True)
    print(f"\n指标已保存: {metrics_path}")

    if result["all_preds"] is not None:
        pred_path = out_dir / "prediction_results.npz"
        np.savez(
            pred_path,
            pred=result["all_preds"],
            target=result["all_targets"],
            future_mask=result["all_future_masks"],
        )
        print(f"预测结果已保存: {pred_path}")

    print("\n预测评估完成。")