# -*- coding: utf-8 -*-
"""设定 A 人工洞 mask 生成：4 形态 × 15%/30%，只挖健康区（obs∧rel）近门。

确定性：seed=1234 + 帧号派生，重复运行完全一致。
输出：bool mask [T, NEAR_GATES, W]，True = 被挖掉（需补全、有真值）。
"""
import numpy as np

from . import cfg


def _scatter(rng, healthy, n_holes):
    idx = np.flatnonzero(healthy)
    pick = rng.choice(idx, size=n_holes, replace=False)
    m = np.zeros_like(healthy)
    m.flat[pick] = True
    return m


def _sector(rng, healthy, n_holes, H, W):
    """方位扇区：连续若干整列方位（环绕 0/360）。"""
    width = max(2, int(round(n_holes / max(H, 1))))
    width = min(width, W)
    start = int(rng.integers(0, W))
    cols = (np.arange(width) + start) % W
    m = np.zeros((H, W), dtype=bool)
    m[:, cols] = True
    m &= healthy
    return _resize_to_n(m, healthy, n_holes, rng)


def _strip(rng, healthy, n_holes, H, W):
    """径向条带：连续若干整行距离门。"""
    height = max(2, int(round(n_holes / max(W, 1))))
    height = min(height, H)
    start = int(rng.integers(0, max(1, H - height + 1)))
    m = np.zeros((H, W), dtype=bool)
    m[start:start + height, :] = True
    m &= healthy
    return _resize_to_n(m, healthy, n_holes, rng)


def _block(rng, healthy, n_holes, H, W):
    """矩形块：随机矩形（可多个直到凑满）。"""
    m = np.zeros((H, W), dtype=bool)
    remaining = n_holes
    for _ in range(50):
        if remaining <= 0:
            break
        bh = int(rng.integers(2, max(3, H // 4)))
        bw = int(rng.integers(2, max(3, W // 4)))
        sh = int(rng.integers(0, max(1, H - bh + 1)))
        sw = int(rng.integers(0, max(1, W - bw + 1)))
        blk = np.zeros((H, W), dtype=bool)
        blk[sh:sh + bh, sw:sw + bw] = True
        add = blk & healthy & ~m
        m |= add
        remaining = n_holes - int(m.sum())
    return _resize_to_n(m, healthy, n_holes, rng)


def _resize_to_n(mask, healthy, n, rng):
    """把 mask∩healthy 调整到恰好 n 个像素。"""
    cand = mask & healthy
    idx = np.flatnonzero(cand)
    if len(idx) > n:
        keep = rng.choice(idx, size=n, replace=False)
        out = np.zeros_like(cand)
        out.flat[keep] = True
        return out
    if len(idx) < n:
        extra_pool = np.flatnonzero(healthy & ~cand)
        need = n - len(idx)
        add = rng.choice(extra_pool, size=min(need, len(extra_pool)), replace=False)
        out = cand.copy()
        out.flat[add] = True
        return out
    return cand


_SHAPE_FN = {
    "scatter": lambda rng, h, n, H, W: _scatter(rng, h, n),
    "sector": _sector,
    "strip": _strip,
    "block": _block,
}


def make_hole_mask_for_frame(obs_near, rel_near, shape, ratio, seed, frame_idx):
    """单帧洞 mask [Hn, W] bool。"""
    healthy = (obs_near > 0.5) & (rel_near > 0.5)
    H, W = healthy.shape
    n_total = int(healthy.sum())
    n_holes = max(1, int(round(ratio * n_total)))
    rng = np.random.default_rng(seed + frame_idx * 1000 + hash(shape) % 1000)
    return _SHAPE_FN[shape](rng, healthy, n_holes, H, W)


def generate_eval_masks(obs, rel, seed=1234):
    """为 test 集生成全部 (shape, ratio) 洞。

    返回 dict[(shape, ratio)] -> bool [T, Hn, W]；并保存到 cfg.EVAL_MASKS_DIR。
    """
    T = obs.shape[0]
    ns, _ = None, None
    obs_near = obs[:, :cfg.NEAR_GATES]
    rel_near = rel[:, :cfg.NEAR_GATES]
    out = {}
    for shape in cfg.HOLE_SHAPES:
        for ratio in cfg.HOLE_RATIOS:
            m = np.zeros((T, cfg.NEAR_GATES, cfg.ORI_W), dtype=bool)
            for t in range(T):
                m[t] = make_hole_mask_for_frame(
                    obs_near[t], rel_near[t], shape, ratio, seed, t)
            out[(shape, ratio)] = m
    return out


def load_or_generate_masks(obs, rel, seed=1234, force=False):
    """带缓存的洞 mask 生成：npz 存 eval_masks/。"""
    import hashlib
    path = cfg.EVAL_MASKS_DIR / "test_holes.npz"
    key = f"seed{seed}_T{obs.shape[0]}_{obs.shape[1]}_{obs.shape[2]}"
    if path.exists() and not force:
        try:
            z = np.load(path, allow_pickle=True)
            if str(z["meta"]) == key:
                out = {}
                for shape in cfg.HOLE_SHAPES:
                    for ratio in cfg.HOLE_RATIOS:
                        k = f"{shape}_{ratio}"
                        out[(shape, ratio)] = z[k].astype(bool)
                return out
        except Exception:
            pass
    out = generate_eval_masks(obs, rel, seed=seed)
    cfg.EVAL_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meta": np.array(key)}
    for (shape, ratio), m in out.items():
        payload[f"{shape}_{ratio}"] = m
    np.savez_compressed(path, **payload)
    return out
