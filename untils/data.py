# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

ORI_H,ORI_W=85,125
PAD_H,PAD_W=91,128
PAD_TOP,PAD_BOTTOM=3,3
PAD_LEFT,PAD_RIGHT=1,2
PHYS_CHANNELS=4
TARGET_CHANNEL=0
BAD_RAYS=[73,74,75]


def pad_to(x,h=PAD_H,w=PAD_W):
    if isinstance(x,torch.Tensor):
        H,W=x.shape[-2:]
        if H==h and W==w:return x
        if H>h or W>w:return x[...,PAD_TOP:PAD_TOP+h,PAD_LEFT:PAD_LEFT+w]
        ph=max(0,h-H);pw=max(0,w-W)
        pt=ph//2;pb=ph-pt
        if ph>0:x=torch.nn.functional.pad(x,(0,0,pt,pb),mode="replicate")
        if pw>0:
            pl=pw//2;pr=pw-pl
            left=x[...,-pl:] if pl>0 else x[...,:0]
            right=x[...,:pr] if pr>0 else x[...,:0]
            x=torch.cat([left,x,right],dim=-1)
        return x[...,:h,:w]
    arr=np.asarray(x)
    H,W=arr.shape[-2:]
    if H==h and W==w:return arr
    if H>h or W>w:return arr[...,PAD_TOP:PAD_TOP+h,PAD_LEFT:PAD_LEFT+w]
    ph=max(0,h-H);pw=max(0,w-W)
    pt=ph//2;pb=ph-pt
    if ph>0:
        arr=np.pad(arr,[(0,0)]*(arr.ndim-2)+[(pt,pb),(0,0)],mode="edge")
    if pw>0:
        pl=pw//2;pr=pw-pl
        if pl>0:
            arr=np.concatenate([arr[...,-pl:],arr],axis=-1)
        if pr>0:
            arr=np.concatenate([arr,arr[...,:pr]],axis=-1)
    return arr[...,:h,:w]


def crop_to(x,h=ORI_H,w=ORI_W):
    if isinstance(x,torch.Tensor):
        return x[...,PAD_TOP:PAD_TOP+h,PAD_LEFT:PAD_LEFT+w]
    x=np.asarray(x)
    return x[...,PAD_TOP:PAD_TOP+h,PAD_LEFT:PAD_LEFT+w]


def _finite_mask(x):
    return np.isfinite(x)


def make_obs_rel(arr,phys_channels=PHYS_CHANNELS):
    x=np.asarray(arr)
    if x.ndim!=4:
        raise ValueError(f"make_obs_rel输入必须为[T,H,W,C]，当前shape={x.shape}")
    T,H,W,C=x.shape
    if C<phys_channels:
        raise ValueError(f"数据通道数不足：C={C}，phys_channels={phys_channels}")
    finite=np.isfinite(x[...,:phys_channels])
    phys_ok=np.all(finite,axis=-1)
    if C>=9:
        # AGENTS.md：通道7=观测掩码(1=有值)，通道8=可靠掩码(1=可靠)
        # 坏射线ch7=0但物理存0.0(有限)，若不看掩码会被误判为有观测→输出透传0
        mask_obs=x[...,7]>0.5
        mask_rel=x[...,8]>0.5
        obs=(mask_obs&phys_ok).astype(np.float32)
        rel=(mask_rel&phys_ok).astype(np.float32)
    else:
        obs=phys_ok.astype(np.float32)
        rel=phys_ok.astype(np.float32)
    for c in range(phys_channels):
        rel[~np.isfinite(x[...,c])]=0.0
    return obs,rel


def random_wrap_interval(start,length,width=ORI_W):
    return ((np.arange(length)+int(start))%width).astype(np.int64)


def _continuous_ray_mask(H,W,n_rays,start=None):
    m=np.zeros((H,W),dtype=np.float32)
    n_rays=max(1,min(int(n_rays),W))
    if start is None:start=np.random.randint(0,W)
    idx=random_wrap_interval(start,n_rays,W)
    m[:,idx]=1.0
    return m


def _radial_dual_mask(H,W):
    m=np.zeros((H,W),dtype=np.float32)
    if H<4:return m
    length=max(2,int(np.random.uniform(0.04,0.15)*H))
    start=np.random.randint(0,max(1,H-length+1))
    m[start:start+length,:]=1.0
    if np.random.rand()<0.6:
        length2=max(2,int(np.random.uniform(0.03,0.10)*H))
        start2=np.random.randint(0,max(1,H-length2+1))
        m[start2:start2+length2,:]=1.0
    return m


def _block_mask(H,W):
    m=np.zeros((H,W),dtype=np.float32)
    bh=max(2,int(np.random.uniform(0.04,0.15)*H))
    bw=max(2,int(np.random.uniform(0.04,0.18)*W))
    bh=min(bh,H);bw=min(bw,W)
    sh=np.random.randint(0,max(1,H-bh+1))
    sw=np.random.randint(0,max(1,W-bw+1))
    m[sh:sh+bh,sw:sw+bw]=1.0
    return m


def _calibrate_missing(m,target_rate,allowed,max_iter=20):
    target=float(np.clip(target_rate,0.0,0.95))
    allowed=allowed.astype(bool)
    n_allowed=int(allowed.sum())
    if n_allowed==0:return np.zeros_like(m,dtype=np.float32)
    target_n=max(1,int(round(target*n_allowed)))
    out=(m>0.5)&allowed
    n=int(out.sum())
    if n==target_n:return out.astype(np.float32)
    if n>target_n:
        idx=np.flatnonzero(out)
        keep=np.random.choice(idx,target_n,replace=False)
        tmp=np.zeros_like(out)
        tmp.flat[keep]=True
        return tmp.astype(np.float32)
    need=target_n-n
    remain=np.flatnonzero(allowed&(~out))
    if len(remain)>0:
        add=np.random.choice(remain,min(need,len(remain)),replace=False)
        out.flat[add]=True
    return out.astype(np.float32)


def _make_one_structural_mask(allowed,target_rate,wedge_min,wedge_max):
    H,W=allowed.shape
    mode=np.random.choice(
        ["azimuth_wedge","radial_dual","block","mixed"],
        p=[0.40,0.20,0.20,0.20]
    )
    if mode=="azimuth_wedge":
        n=np.random.randint(wedge_min,wedge_max+1)
        m=_continuous_ray_mask(H,W,n)
    elif mode=="radial_dual":
        m=_radial_dual_mask(H,W)
    elif mode=="block":
        m=_block_mask(H,W)
    else:
        m=np.maximum(_continuous_ray_mask(H,W,np.random.randint(wedge_min,wedge_max+1)),_block_mask(H,W))
        if np.random.rand()<0.5:
            m=np.maximum(m,_radial_dual_mask(H,W))
    return _calibrate_missing(m,target_rate,allowed)


def _legacy_structural_mask(B,H,W,device,drop_prob,cfg=None,valid_mask=None):
    if valid_mask is None:
        allowed=np.ones((B,H,W),dtype=np.float32)
    else:
        vm=valid_mask.detach().float().cpu().numpy()
        if vm.ndim==5:vm=vm[:,0,0]
        elif vm.ndim==4:vm=vm[:,0]
        allowed=vm
    target=float(getattr(cfg,"artificial_missing_rate",0.06) if cfg is not None else 0.06)
    if drop_prob is not None:
        target=float(getattr(cfg,"drop_prob",target))
        if target>0.30:target=0.06
    out=np.zeros((B,H,W),dtype=np.float32)
    rates=[]
    for b in range(B):
        out[b]=_make_one_structural_mask(
            allowed[b]>0.5,target,
            int(getattr(cfg,"wedge_min",2) if cfg is not None else 2),
            int(getattr(cfg,"wedge_max",6) if cfg is not None else 6)
        )
        rates.append(float(out[b].sum()/max((allowed[b]>0.5).sum(),1)))
    return torch.from_numpy(out[:,None]).float().to(device),rates


def generate_structural_missing_mask(obs,rel=None,target_rate=0.06,wedge_min=2,wedge_max=6,p_mid=0.01,**kwargs):
    """
    当前接口：
        generate_structural_missing_mask(obs,rel,target_rate=0.06,...)
        obs/rel: [T,H,W]
    兼容旧train接口：
        generate_structural_missing_mask(B,H,W,device,drop_prob,cfg=...,valid_mask=...)
    """
    if isinstance(obs,(int,np.integer)) and isinstance(rel,(int,np.integer)):
        B,H,W=obs,int(rel),int(target_rate)
        device=wedge_min
        drop_prob=wedge_max
        cfg=kwargs.get("cfg",None)
        valid_mask=kwargs.get("valid_mask",None)
        return _legacy_structural_mask(B,H,W,device,drop_prob,cfg,valid_mask)

    obs=np.asarray(obs,dtype=np.float32)
    rel=np.asarray(rel,dtype=np.float32)
    if obs.ndim!=3 or rel.shape!=obs.shape:
        raise ValueError(f"obs/rel必须都是[T,H,W]，当前obs={obs.shape},rel={rel.shape}")
    T,H,W=obs.shape
    out=np.zeros((T,H,W),dtype=np.float32)
    for t in range(T):
        allowed=(obs[t]>0.5)&(rel[t]>0.5)
        out[t]=_make_one_structural_mask(
            allowed,float(target_rate),int(wedge_min),int(wedge_max)
        )
        if p_mid>0 and np.random.rand()<float(p_mid):
            # 仅作为极低概率的小块扰动，不形成主导缺失模式
            extra=_block_mask(H,W)*allowed
            out[t]=np.maximum(out[t],extra)
            out[t]=_calibrate_missing(out[t],target_rate,allowed)
    return out.astype(np.float32)


def _pure_structural_mask(allowed, drop_prob):
    H, W = allowed.shape
    n_wedge = max(1, int(drop_prob * 10))
    n_far_wedge = max(2, int(drop_prob * 25))
    far_ratio = np.random.uniform(0.3, 0.7)

    m = np.zeros((H, W), dtype=np.float32)

    wedge_start = np.random.randint(0, W)
    wedge_idx = random_wrap_interval(wedge_start, n_wedge, W)
    m[:, wedge_idx] = 1.0

    far_start_gate = int(H * (1.0 - far_ratio))
    far_start = np.random.randint(0, W)
    far_idx = random_wrap_interval(far_start, n_far_wedge, W)
    m[far_start_gate:, far_idx] = 1.0

    return m * allowed.astype(np.float32)


def generate_pure_structural_missing_mask(B, H, W, device, drop_prob, cfg=None, valid_mask=None):
    if valid_mask is None:
        allowed = np.ones((B, H, W), dtype=np.float32)
    else:
        vm = valid_mask.detach().float().cpu().numpy()
        if vm.ndim == 5:
            vm = vm[:, 0, 0]
        elif vm.ndim == 4:
            vm = vm[:, 0]
        allowed = vm

    out = np.zeros((B, H, W), dtype=np.float32)
    rates = []
    for b in range(B):
        out[b] = _pure_structural_mask(allowed[b] > 0.5, drop_prob)
        rates.append(float(out[b].sum() / max((allowed[b] > 0.5).sum(), 1)))

    return torch.from_numpy(out[:, None]).float().to(device), rates


def make_effective_mask(obs,rel,art=None):
    obs=np.asarray(obs,dtype=np.float32)
    rel=np.asarray(rel,dtype=np.float32)
    m=obs*rel
    if art is not None:m=m*(1.0-np.asarray(art,dtype=np.float32))
    return m.astype(np.float32)


class Normalizer:
    def __init__(self,method="minmax",clip_percentile=99.5,num_channels=PHYS_CHANNELS):
        self.method=method
        self.clip_percentile=clip_percentile
        self.num_channels=num_channels
        self.lo=np.zeros(num_channels,dtype=np.float64)
        self.hi=np.ones(num_channels,dtype=np.float64)
        self.mean=np.zeros(num_channels,dtype=np.float64)
        self.std=np.ones(num_channels,dtype=np.float64)

    def fit(self,arr,obs):
        arr=np.asarray(arr)
        obs=np.asarray(obs,dtype=bool)
        vals=arr[...,:self.num_channels]
        for c in range(self.num_channels):
            v=vals[...,c][obs]
            v=v[np.isfinite(v)].astype(np.float64)
            if len(v)==0:raise ValueError(f"通道{c}没有有效观测")
            if self.method=="minmax":
                lo,hi=np.percentile(v,[100-self.clip_percentile,self.clip_percentile])
                if hi<=lo:hi=lo+1e-6
                self.lo[c],self.hi[c]=lo,hi
            else:
                self.mean[c]=v.mean()
                self.std[c]=v.std()+1e-8
        return self

    def transform(self,arr):
        x=np.asarray(arr,dtype=np.float32).copy()
        out=np.zeros_like(x,dtype=np.float32)
        for c in range(min(self.num_channels,x.shape[-1])):
            if self.method=="minmax":
                out[...,c]=(x[...,c]-self.lo[c])/max(self.hi[c]-self.lo[c],1e-8)*2.0-1.0
            else:
                out[...,c]=(x[...,c]-self.mean[c])/self.std[c]
        return out

    def inverse_transform(self,arr):
        x=np.asarray(arr,dtype=np.float32).copy()
        out=x.copy()
        for c in range(min(self.num_channels,x.shape[-1])):
            if self.method=="minmax":
                out[...,c]=(x[...,c]+1.0)*0.5*(self.hi[c]-self.lo[c])+self.lo[c]
            else:
                out[...,c]=x[...,c]*self.std[c]+self.mean[c]
        return out

    def save(self,path):
        np.savez(
            path,method=self.method,clip_percentile=self.clip_percentile,
            num_channels=self.num_channels,lo=self.lo,hi=self.hi,
            mean=self.mean,std=self.std
        )

    def load(self,path):
        z=np.load(path,allow_pickle=True)
        self.method=str(z["method"])
        self.clip_percentile=float(z["clip_percentile"])
        self.num_channels=int(z["num_channels"])
        self.lo=z["lo"]
        self.hi=z["hi"]
        self.mean=z["mean"]
        self.std=z["std"]
        return self


class MultichannelClipDataset(Dataset):
    def __init__(self,npy_path,normalizer=None,time_steps=1,phys_channels=PHYS_CHANNELS,return_meta=False):
        self.npy_path=npy_path
        self.arr=np.load(npy_path).astype(np.float32)
        if self.arr.ndim!=4:
            raise ValueError(f"数据必须为[T,H,W,C]，当前shape={self.arr.shape}")
        self.time_steps=int(time_steps)
        self.phys_channels=int(phys_channels)
        self.return_meta=return_meta
        self.normalizer=normalizer
        self.obs,self.rel=make_obs_rel(self.arr,self.phys_channels)
        self.length=max(1,self.arr.shape[0]-self.time_steps+1)

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        idx=int(idx)
        end=min(idx+self.time_steps,self.arr.shape[0])
        x=self.arr[idx:end,...,:self.phys_channels]
        obs=self.obs[idx:end]
        rel=self.rel[idx:end]
        if end-idx<self.time_steps:
            n=self.time_steps-(end-idx)
            x=np.concatenate([x,np.repeat(x[-1:],n,axis=0)],axis=0)
            obs=np.concatenate([obs,np.repeat(obs[-1:],n,axis=0)],axis=0)
            rel=np.concatenate([rel,np.repeat(rel[-1:],n,axis=0)],axis=0)
        if self.normalizer is not None:
            x0=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
            x=self.normalizer.transform(x0)
        x=np.where(obs[...,None]>0.5,x,0.0).astype(np.float32)
        x=np.transpose(x,(3,0,1,2))
        m=np.repeat(obs[None,...],self.phys_channels,axis=0).astype(np.float32)
        r=np.repeat(rel[None,...],self.phys_channels,axis=0).astype(np.float32)
        x=pad_to(x)
        m=pad_to(m)
        r=pad_to(r)
        v=np.repeat(rel[None,...],self.phys_channels,axis=0).astype(np.float32)
        v=pad_to(v)
        return torch.from_numpy(x),torch.from_numpy(m),torch.from_numpy(r),torch.from_numpy(v)


def load_dataset(train_path,val_path,test_path,normalizer=None,time_steps=1,phys_channels=PHYS_CHANNELS):
    train_ds=MultichannelClipDataset(train_path,normalizer,time_steps,phys_channels)
    val_ds=MultichannelClipDataset(val_path,normalizer,time_steps,phys_channels)
    test_ds=MultichannelClipDataset(test_path,normalizer,time_steps,phys_channels)
    return train_ds,val_ds,test_ds


def set_bad_rays(mask,ray_indices=BAD_RAYS,value=0.0):
    if isinstance(mask,torch.Tensor):
        m=mask.clone()
        for ray in ray_indices:
            j=int(ray)+PAD_LEFT
            if 0<=j<m.shape[-1]:
                m[...,j]=value
        return m
    m=np.asarray(mask).copy()
    for ray in ray_indices:
        j=int(ray)+PAD_LEFT
        if 0<=j<m.shape[-1]:
            m[...,j]=value
    return m


def apply_bad_rays(x,ray_indices=BAD_RAYS,value=0.0):
    if isinstance(x,torch.Tensor):
        y=x.clone()
        for ray in ray_indices:
            j=int(ray)+PAD_LEFT
            if 0<=j<y.shape[-1]:
                y[...,j]=value
        return y
    y=np.asarray(x).copy()
    for ray in ray_indices:
        j=int(ray)+PAD_LEFT
        if 0<=j<y.shape[-1]:
            y[...,j]=value
    return y


def make_obs_rel_padded(arr,phys_channels=PHYS_CHANNELS):
    obs,rel=make_obs_rel(arr,phys_channels)
    return pad_to(obs),pad_to(rel)


def normalize_and_pad(arr,normalizer,phys_channels=PHYS_CHANNELS):
    obs,rel=make_obs_rel(arr,phys_channels)
    x=np.nan_to_num(arr[...,:phys_channels].astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    if normalizer is not None:x=normalizer.transform(x)
    x=np.where(obs[...,None]>0.5,x,0.0)
    x=np.transpose(x,(3,0,1,2))
    return pad_to(x),pad_to(obs),pad_to(rel)


class SlidingWindowDataset(Dataset):
    """滑动窗口数据集：将长序列按时间切分为 (历史, 未来) 窗口。

    从时间序列中提取滑动窗口，每个窗口包含：
      - 历史帧：t 到 t+input_steps-1
      - 未来帧：t+input_steps 到 t+input_steps+pred_steps-1

    先按时间切分（train/val/test 来自不同文件），再在各自序列内生成窗口。
    窗口不会跨越时间断档（由文件边界保证）。

    返回：
        history_x    : [phys_channels, input_steps, PAD_H, PAD_W]
        history_m    : [phys_channels, input_steps, PAD_H, PAD_W]
        history_r    : [phys_channels, input_steps, PAD_H, PAD_W]
        future_x     : [phys_channels, pred_steps, PAD_H, PAD_W]
        future_m     : [phys_channels, pred_steps, PAD_H, PAD_W]
        future_r     : [phys_channels, pred_steps, PAD_H, PAD_W]
    """

    def __init__(self, npy_path, normalizer=None, input_steps=10,
                 pred_steps=1, phys_channels=PHYS_CHANNELS):
        self.npy_path = npy_path
        self.arr = np.load(npy_path).astype(np.float32)
        if self.arr.ndim != 4:
            raise ValueError(
                f"数据必须为 [T,H,W,C]，当前 shape={self.arr.shape}"
            )
        self.input_steps = int(input_steps)
        self.pred_steps = int(pred_steps)
        self.phys_channels = int(phys_channels)
        self.normalizer = normalizer
        self.obs, self.rel = make_obs_rel(self.arr, self.phys_channels)

        total_window = self.input_steps + self.pred_steps
        self.length = max(1, self.arr.shape[0] - total_window + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = int(idx)
        total = self.input_steps + self.pred_steps
        end = min(idx + total, self.arr.shape[0])

        x_all = self.arr[idx:end, ..., :self.phys_channels]
        obs_all = self.obs[idx:end]
        rel_all = self.rel[idx:end]

        if end - idx < total:
            need = total - (end - idx)
            x_all = np.concatenate(
                [x_all, np.repeat(x_all[-1:], need, axis=0)], axis=0
            )
            obs_all = np.concatenate(
                [obs_all, np.repeat(obs_all[-1:], need, axis=0)], axis=0
            )
            rel_all = np.concatenate(
                [rel_all, np.repeat(rel_all[-1:], need, axis=0)], axis=0
            )

        history_x = x_all[:self.input_steps]
        future_x = x_all[self.input_steps:self.input_steps + self.pred_steps]
        history_obs = obs_all[:self.input_steps]
        future_obs = obs_all[self.input_steps:self.input_steps + self.pred_steps]
        history_rel = rel_all[:self.input_steps]
        future_rel = rel_all[self.input_steps:self.input_steps + self.pred_steps]

        if self.normalizer is not None:
            hx = np.nan_to_num(history_x, nan=0.0, posinf=0.0, neginf=0.0)
            fx = np.nan_to_num(future_x, nan=0.0, posinf=0.0, neginf=0.0)
            history_x = self.normalizer.transform(hx)
            future_x = self.normalizer.transform(fx)

        history_x = np.where(
            history_obs[..., None] > 0.5, history_x, 0.0
        ).astype(np.float32)
        future_x = np.where(
            future_obs[..., None] > 0.5, future_x, 0.0
        ).astype(np.float32)

        history_x = np.transpose(history_x, (3, 0, 1, 2))
        future_x = np.transpose(future_x, (3, 0, 1, 2))

        history_m = np.repeat(
            history_obs[None, ...], self.phys_channels, axis=0
        ).astype(np.float32)
        future_m = np.repeat(
            future_obs[None, ...], self.phys_channels, axis=0
        ).astype(np.float32)
        history_r = np.repeat(
            history_rel[None, ...], self.phys_channels, axis=0
        ).astype(np.float32)
        future_r = np.repeat(
            future_rel[None, ...], self.phys_channels, axis=0
        ).astype(np.float32)

        history_x = pad_to(history_x)
        history_m = pad_to(history_m)
        history_r = pad_to(history_r)
        future_x = pad_to(future_x)
        future_m = pad_to(future_m)
        future_r = pad_to(future_r)

        return (
            torch.from_numpy(history_x),
            torch.from_numpy(history_m),
            torch.from_numpy(history_r),
            torch.from_numpy(future_x),
            torch.from_numpy(future_m),
            torch.from_numpy(future_r),
        )


class RangeExtensionSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path, normalizer=None, phys_channels=PHYS_CHANNELS,
                 near_end=68, far_end=78, temporal_steps=3):
        self.arr = np.load(npy_path).astype(np.float32)
        if self.arr.ndim != 4:
            raise ValueError(f"数据必须为[T,H,W,C]，当前shape={self.arr.shape}")
        self.normalizer = normalizer
        self.phys_channels = phys_channels
        self.near_end = near_end
        self.far_end = far_end
        self.temporal_steps = temporal_steps
        self.obs, self.rel = make_obs_rel(self.arr, phys_channels)
        self.length = max(1, self.arr.shape[0] - temporal_steps + 1)

    def __len__(self):
        return self.length

    def _process_frame(self, idx):
        x = self.arr[idx]
        obs = self.obs[idx]
        rel = self.rel[idx]
        near_raw = x[:self.near_end, :, :self.phys_channels]
        far_raw = x[self.near_end:self.far_end, :, :self.phys_channels]
        near_obs = obs[:self.near_end]
        far_obs = obs[self.near_end:self.far_end]
        near_rel = rel[:self.near_end]
        far_rel = rel[self.near_end:self.far_end]

        if self.normalizer is not None:
            near_raw = np.nan_to_num(near_raw, nan=0.0, posinf=0.0, neginf=0.0)
            near_raw = self.normalizer.transform(near_raw)
            far_raw = np.nan_to_num(far_raw, nan=0.0, posinf=0.0, neginf=0.0)
            far_raw = self.normalizer.transform(far_raw)
        else:
            near_raw = np.nan_to_num(near_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            far_raw = np.nan_to_num(far_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        near_phys = np.where(near_obs[..., None] > 0.5, near_raw, 0.0).astype(np.float32)
        # 修复1：将 [H_near, W, C] 转置为 [C, H_near, W]
        near_phys = np.transpose(near_phys, (2, 0, 1))
        near_obs_ch = near_obs[None, ...].astype(np.float32)
        near_rel_ch = near_rel[None, ...].astype(np.float32)
        near_x = np.concatenate([near_phys, near_obs_ch, near_rel_ch], axis=0)

        # 修复2：far_v 保持 [H_far, W]，并输出准确的 far_valid 掩码
        far_v = far_raw[..., 0].astype(np.float32)
        far_v = np.where(far_obs > 0.5, far_v, 0.0).astype(np.float32)
        far_valid = (far_obs * far_rel).astype(np.float32)
        return near_x, far_v, far_valid

    def __getitem__(self, idx):
        idx = int(idx)
        end = min(idx + self.temporal_steps, self.arr.shape[0])
        seq_list = []
        for t in range(idx, end):
            nx, _, _ = self._process_frame(t)
            seq_list.append(nx)
        if len(seq_list) < self.temporal_steps:
            seq_list += [seq_list[-1]] * (self.temporal_steps - len(seq_list))
        mid_idx = min(idx + (self.temporal_steps - 1) // 2, self.arr.shape[0] - 1)
        x_cur, y_far, far_valid = self._process_frame(mid_idx)
        x_seq = np.stack(seq_list, axis=0)
        return (
            torch.from_numpy(x_cur),
            torch.from_numpy(x_seq),
            torch.from_numpy(y_far),
            torch.from_numpy(far_valid),
        )