# -*- coding:utf-8 -*-
import csv,random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from untils.data import MultichannelClipDataset,Normalizer,generate_pure_structural_missing_mask,make_obs_rel,pad_to,crop_to,ORI_H,ORI_W,PAD_H,PAD_W,PAD_TOP,PAD_LEFT
from untils.losses import azimuth_curvature_loss,azimuth_tv_loss,discriminator_loss,generate_hint,generator_adv_loss,reconstruction_loss,temporal_consistency_loss,tv_loss
from model import UNetGenerator,PatchGANDiscriminator,init_weights
from untils.exp_utils import print_model_stats
from untils.losses_refine import compute_refiner_losses
from untils.refiner import ResidualRefiner,apply_refiner
from untils.wavelet import wavelet_feature_channels

BAD_RAYS=[73,74,75]
CHANNEL_NAMES=["V","CNR","SW","Skew"]
TARGET_CHANNEL=0

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def fit_normalizer(train_npy,method,clip,phys_channels):
    arr=np.load(train_npy);obs,_=make_obs_rel(arr,phys_channels)
    return Normalizer(method,clip,phys_channels).fit(arr,obs)

def repeat_mask(m,channels):
    if m.dim()==4:m=m.unsqueeze(1)
    return m.repeat(1,channels,1,1,1) if m.shape[1]==1 else m

def generate_contextual_cue(x,effective_mask,target_t,target_channel=0):
    B,C,T,H,W=x.shape;v=x[:,target_channel:target_channel+1]
    if effective_mask.dim()==4:effective_mask=effective_mask.unsqueeze(1)
    mask_v=effective_mask[:,target_channel:target_channel+1]
    ti=torch.arange(T,device=x.device,dtype=torch.float32)
    dist=torch.abs(ti-float(target_t)).view(1,T,1,1).expand(B,T,H,W).clone()
    valid=mask_v[:,0]>0.5;dist[~valid]=float("inf")
    nt=torch.argmin(dist,dim=1);md=torch.amin(dist,dim=1)
    nv=torch.gather(v[:,0],1,nt.unsqueeze(1));hc=torch.isfinite(md)
    return torch.where(hc.unsqueeze(1),nv,torch.zeros_like(nv))

def build_contextual_fusion(x,effective_mask,cue,z,noise_scale=.1):
    B,C,T,H,W=x.shape;m=repeat_mask(effective_mask,C);xb=x.clone();mv=m[:,0:1]
    cue5=cue.unsqueeze(2).expand(-1,-1,T,-1,-1);zv=z[:,0:1]
    xb[:,0:1]=mv*x[:,0:1]+(1-mv)*cue5+(1-mv)*zv*noise_scale
    if C>1:xb[:,1:]=m[:,1:]*x[:,1:]+(1-m[:,1:])*z[:,1:]*noise_scale
    return xb

def build_generator_inputs(x_bar,m):
    m=repeat_mask(m,x_bar.shape[1])
    if not hasattr(build_generator_inputs,"_printed"):
        print("Generator输入:",f"x_bar={tuple(x_bar.shape)}",f"mask={tuple(m.shape)}",f"total={x_bar.shape[1]+m.shape[1]}")
        build_generator_inputs._printed=True
    return x_bar,m

def make_training_cue(x,m,r,valid,target_t):
    m5=repeat_mask(m,x.shape[1]);r5=repeat_mask(r,x.shape[1]);v5=repeat_mask(valid,x.shape[1])
    em=m5*r5*v5
    return generate_contextual_cue(x,em,target_t,TARGET_CHANNEL),em

def build_dataset(path,norm,cfg):
    return MultichannelClipDataset(path,normalizer=norm,time_steps=cfg.time_steps,phys_channels=cfg.phys_channels)

def build_generator(cfg):
    return UNetGenerator(in_channels=8,out_channels=1,base_filter=cfg.base_filter,use_cue=getattr(cfg,'use_cue',True),use_nafblock=getattr(cfg,'use_nafblock',True),use_volformer=getattr(cfg,'use_volformer',False),use_axial_w=getattr(cfg,'use_axial_w',True),axial_num_pairs=getattr(cfg,'axial_num_pairs',4),axial_num_heads=getattr(cfg,'axial_num_heads',4))

def build_refiner(cfg):
    if not getattr(cfg,"use_refiner",False):return None
    phys_ch=int(getattr(cfg,"phys_channels",4))
    geom_ch=3;mask_ch=2
    base_ch=phys_ch+geom_ch+mask_ch
    if getattr(cfg,"use_wavelet",False):
        wl=wavelet_feature_channels(int(getattr(cfg,"n_wavelet_levels",1)))
    else:wl=0
    in_ch=base_ch+wl
    return ResidualRefiner(in_channels=in_ch,base_filter=cfg.refiner_base_filter)

def prepare_artificial_mask(B,H,W,device,cfg,valid_eff):
    art,rates=generate_pure_structural_missing_mask(B,H,W,device,cfg.drop_prob,cfg=cfg,valid_mask=valid_eff)
    return (art if art.dim()==4 else art.unsqueeze(1)).float(),rates

def train_one_epoch(G,D,R,loader,optim_G,optim_D,cfg,device):
    G.train();D.train()
    if R is not None:R.train()
    keys=["G","D","D_real","D_fake","G_adv","G_rec","TV","RING","AZTV","CURV","TEMP","REFINE","WAVE","PHY","train_miss_rate","real_missing_rate","total_missing_rate"]
    s={k:0. for k in keys};n=0
    for batch in tqdm(loader,desc="train",leave=False):
        x,m,r,valid=batch;x,m,r,valid=[v.to(device) for v in (x,m,r,valid)]
        B,C,T,H,W=x.shape;t=min(int(cfg.target_idx),T-1)
        vobs=m[:,0:1,t];vrel=r[:,0:1,t];vval=valid[:,0:1,t];ve=vrel*vval
        art,rates=prepare_artificial_mask(B,H,W,device,cfg,ve);art=art*ve
        if isinstance(rates,list):rv=float(np.mean(rates))
        elif torch.is_tensor(rates):rv=float(rates.float().mean().item())
        else:rv=float(rates)
        s["train_miss_rate"]+=rv
        obs_all=m[:,:,t];rel_all=r[:,:,t];valid_all=valid[:,:,t]
        given_all=obs_all*(1-art);vgiven=vobs*(1-art)
        real_missing=(1-obs_all)*valid_all;total_missing=(1-vgiven)*vval
        s["real_missing_rate"]+=(real_missing.sum()/valid_all.sum().clamp_min(1.)).item()
        s["total_missing_rate"]+=(total_missing.sum()/vval.sum().clamp_min(1.)).item()
        xin=x.clone();xin[:,:,t]=x[:,:,t]*given_all;min_=m.clone();min_[:, :, t]=given_all;z=torch.randn_like(xin)
        cue,em=make_training_cue(xin,min_,r,valid,t)
        if not cfg.use_cue:cue=torch.zeros(B,1,H,W,device=device,dtype=x.dtype)
        xb=build_contextual_fusion(xin,em,cue,z,cfg.noise_scale);gx,gm=build_generator_inputs(xb,em)
        vf=G(gx,gm);vfake=vf[:,0,t:t+1];vtrue=x[:,0,t:t+1];gv=vgiven
        vhat=vtrue*gv+vfake*(1-gv);hint=generate_hint(gv,cfg.hint_rate);dl=D(vhat.detach(),hint);d_loss=discriminator_loss(dl,gv,vval)
        optim_D.zero_grad(set_to_none=True);d_loss.backward()
        if cfg.grad_clip>0:torch.nn.utils.clip_grad_norm_(D.parameters(),cfg.grad_clip)
        optim_D.step()
        vf=G(gx,gm);vfake=vf[:,0,t:t+1];vhat=vtrue*gv+vfake*(1-gv);dlg=D(vhat,hint)
        g_adv=generator_adv_loss(dlg,art,vval);g_rec=reconstruction_loss(vfake,vtrue,art,vrel)
        g_tv=tv_loss(vfake,art,vval) if cfg.tv_weight>0 else x.new_zeros(())
        g_temp=temporal_consistency_loss(vf,t,art,vval) if cfg.temp_weight>0 and T>1 else x.new_zeros(())
        g_ring=(crop_to(vfake,ORI_H,ORI_W)[...,0]-crop_to(vfake,ORI_H,ORI_W)[...,-1]).abs().mean() if cfg.ring_weight>0 else x.new_zeros(())
        mask_az=torch.clamp(art+(1-vrel)*vval,0,1)
        g_aztv=azimuth_tv_loss(vfake,mask_az) if cfg.aztv_weight>0 else x.new_zeros(())
        g_curv=azimuth_curvature_loss(vfake,mask_az) if cfg.curv_weight>0 else x.new_zeros(())
        gt=(cfg.adv_weight*g_adv+cfg.alpha*cfg.recon_weight*g_rec+cfg.tv_weight*g_tv+cfg.temp_weight*g_temp+cfg.ring_weight*g_ring+cfg.aztv_weight*g_aztv+cfg.curv_weight*g_curv)
        rv=wvl=phy=0.
        if R is not None and getattr(cfg,"use_refiner",False):
            try:
                _,GR,GW,GP,_,_=compute_refiner_losses(R,vfake,x,obs_all,vgiven,vrel,vval,vtrue,art,t,cfg,device)
                gt=gt+cfg.lambda_refine*GR+cfg.lambda_wavelet*GW+cfg.lambda_physics*GP
                rv,wvl,phy=GR.item(),GW.item(),GP.item()
            except Exception as e:print("Refiner本轮跳过：",e)
        optim_G.zero_grad(set_to_none=True);gt.backward()
        if cfg.grad_clip>0:
            torch.nn.utils.clip_grad_norm_(G.parameters(),cfg.grad_clip)
            if R is not None:torch.nn.utils.clip_grad_norm_(R.parameters(),cfg.grad_clip)
        optim_G.step()
        with torch.no_grad():
            p=torch.sigmoid(dl);op=F.interpolate(gv,size=p.shape[-2:],mode="nearest");mp=F.interpolate(art,size=p.shape[-2:],mode="nearest");vp=F.interpolate(vval,size=p.shape[-2:],mode="nearest")
            rw=op*vp;fw=mp*vp;dr=(p*rw).sum().item()/(rw.sum().item()+1e-8);df=(p*fw).sum().item()/(fw.sum().item()+1e-8)
        vals={"G":gt.item(),"D":d_loss.item(),"D_real":dr,"D_fake":df,"G_adv":g_adv.item(),"G_rec":g_rec.item(),"TV":g_tv.item(),"RING":g_ring.item(),"AZTV":g_aztv.item(),"CURV":g_curv.item(),"TEMP":g_temp.item(),"REFINE":rv,"WAVE":wvl,"PHY":phy}
        for k,v in vals.items():s[k]+=v
        n+=1
    return {k:v/max(n,1) for k,v in s.items()}

@torch.no_grad()
def validate_artificial(G,R,loader,cfg,device):
    G.eval()
    if R is not None:R.eval()
    maes=mse=cnt=rs=rn=0.
    for batch in loader:
        x,m,r,valid=batch;x,m,r,valid=[v.to(device) for v in (x,m,r,valid)]
        B,C,T,H,W=x.shape;t=min(int(cfg.target_idx),T-1)
        vobs=m[:,0:1,t];vrel=r[:,0:1,t];vval=valid[:,0:1,t];ve=vrel*vval
        art,rates=prepare_artificial_mask(B,H,W,device,cfg,ve);art=art*ve;vgiven=vobs*(1-art);given_all=m[:,:,t]*(1-art)
        if isinstance(rates,list):rs+=float(np.sum(rates));rn+=len(rates)
        elif torch.is_tensor(rates):rs+=float(rates.float().sum().item());rn+=rates.numel()
        else:rs+=float(rates);rn+=1
        xin=x.clone();xin[:,:,t]=x[:,:,t]*given_all;min_=m.clone();min_[:, :, t]=given_all;z=torch.zeros_like(xin)
        cue,em=make_training_cue(xin,min_,r,valid,t)
        if not cfg.use_cue:cue=torch.zeros(B,1,H,W,device=device,dtype=x.dtype)
        xb=build_contextual_fusion(xin,em,cue,z,0.);gx,gm=build_generator_inputs(xb,em);pred=G(gx,gm)[:,0,t:t+1]
        if R is not None and getattr(cfg,"use_refiner",False):
            try:pred=apply_refiner(R,pred,x,vgiven,vrel,vval,cfg,device)
            except Exception:pass
        mask=(art*vrel*vval);err=pred-x[:,0,t:t+1];maes+=(err.abs()*mask).sum().item();mse+=((err**2)*mask).sum().item();cnt+=mask.sum().item()
    return maes/max(cnt,1.),np.sqrt(mse/max(cnt,1.)),rs/max(rn,1)

@torch.no_grad()
def fixed_ray_surrogate(G,R,npy_path,normalizer,cfg,device,bad_rays=BAD_RAYS):
    arr=np.load(npy_path);obs,rel=make_obs_rel(arr,cfg.phys_channels)
    if arr.ndim!=4:return None
    T,H,W,C=arr.shape;bad=[int(v) for v in bad_rays if 0<=int(v)<W]
    if not bad:return None
    ta=tsq=0.;tn=0
    phys=np.nan_to_num(arr[...,:cfg.phys_channels],nan=0.,posinf=0.,neginf=0.).astype(np.float32);norm=normalizer.transform(phys)
    for i in range(T):
        xn=pad_to(norm[i:i+1].transpose(3,0,1,2));mn=pad_to(obs[i:i+1]);rn=pad_to(rel[i:i+1])
        xt=torch.from_numpy(xn.copy()).float().unsqueeze(0).to(device);mt=torch.from_numpy(mn.copy()).float().unsqueeze(1).to(device);rt=torch.from_numpy(rn.copy()).float().unsqueeze(1).to(device);vt=torch.ones_like(mt)
        for br in bad:
            j=br+PAD_LEFT
            if 0<=j<PAD_W:mt[:,:,:,j]=0.;xt[:,:,0,:,j]=0.
        em=mt*rt*vt;z=torch.zeros_like(xt);cue=generate_contextual_cue(xt,em,0,TARGET_CHANNEL);xb=build_contextual_fusion(xt,em,cue,z,0.);gx,gm=build_generator_inputs(xb,em);pred=G(gx,gm)[:,:,0]
        if R is not None and getattr(cfg,"use_refiner",False):
            try:pred=apply_refiner(R,pred,xt,mt[:,:,0],rt[:,:,0],vt[:,:,0],cfg,device)
            except Exception:pass
        pv=crop_to(pred,ORI_H,ORI_W)[0,0].cpu().numpy();gv=norm[i,:,:,TARGET_CHANNEL]
        bm=np.zeros((H,W),bool);bm[:,bad]=True;mask=bm
        if mask.any():
            e=pv[mask]-gv[mask];ta+=np.abs(e).sum();tsq+=(e**2).sum();tn+=int(mask.sum())
    return {"V_MAE_norm":ta/max(tn,1),"V_RMSE_norm":np.sqrt(tsq/max(tn,1)),"points":tn,"bad_rays":bad}

def save_history(history,out_dir):
    if not history:return
    out_dir=Path(out_dir);keys=list(history[0].keys())
    with open(out_dir/"training_history.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(history)
    ep=[h["epoch"] for h in history]
    plt.figure(figsize=(10,6));plt.plot(ep,[h["G"] for h in history],label="G");plt.plot(ep,[h["D"] for h in history],label="D");plt.xlabel("Epoch");plt.ylabel("Loss");plt.title("G / D Loss");plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(out_dir/"loss_G_D.png",dpi=180);plt.close()
    plt.figure(figsize=(10,6));plt.plot(ep,[h["D_real"] for h in history],label="D(real)");plt.plot(ep,[h["D_fake"] for h in history],label="D(fake)");plt.axhline(.5,ls="--",lw=1);plt.xlabel("Epoch");plt.ylabel("Probability");plt.title("D(real) / D(fake)");plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(out_dir/"D_real_fake.png",dpi=180);plt.close()
    plt.figure(figsize=(10,6));plt.plot(ep,[h["V_MAE"] for h in history],label="V MAE");plt.plot(ep,[h["V_RMSE"] for h in history],label="V RMSE");plt.xlabel("Epoch");plt.ylabel("Error");plt.title("Artificial Missing V Error");plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(out_dir/"wind_speed_metrics.png",dpi=180);plt.close()
    plt.figure(figsize=(10,6));plt.plot(ep,[h.get("train_miss_rate",0) for h in history],label="Train artificial");plt.plot(ep,[h.get("val_miss_rate",0) for h in history],label="Val artificial");plt.plot(ep,[h.get("real_missing_rate",0) for h in history],label="Real missing");plt.plot(ep,[h.get("total_missing_rate",0) for h in history],label="Total missing");plt.axhline(.06,ls="--",lw=1,label="Target 6%");plt.xlabel("Epoch");plt.ylabel("Missing Rate");plt.title("Missing Rate");plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(out_dir/"missing_rate.png",dpi=180);plt.close()

def main():
    cfg=Config();set_seed(cfg.seed);device=torch.device(cfg.device);out_dir=Path(cfg.output_dir);out_dir.mkdir(parents=True,exist_ok=True)
    print(f"设备={device},time_steps={cfg.time_steps},batch={cfg.batch_size}");print(f"原始尺寸={ORI_H}x{ORI_W},模型尺寸={PAD_H}x{PAD_W}");print("目标：只生成V，CNR/SW/Skew作为条件输入。")
    norm=fit_normalizer(cfg.train_npy,cfg.normalize,cfg.clip_percentile,cfg.phys_channels)
    train_ds=build_dataset(cfg.train_npy,norm,cfg);val_ds=build_dataset(cfg.val_npy,norm,cfg)
    train_loader=DataLoader(train_ds,batch_size=cfg.batch_size,shuffle=True,drop_last=True,num_workers=0)
    val_loader=DataLoader(val_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=0)
    G=build_generator(cfg).to(device);D=PatchGANDiscriminator(2,16).to(device);init_weights(G);init_weights(D);R=build_refiner(cfg)
    if R is not None:R=R.to(device);init_weights(R)
    try:print_model_stats(G,D,R) if R is not None else print_model_stats(G,D)
    except Exception:pass
    gp=list(G.parameters())+(list(R.parameters()) if R is not None else [])
    optim_G=torch.optim.Adam(gp,lr=getattr(cfg,'lr_g',getattr(cfg,'lr',2e-4)),betas=cfg.betas);optim_D=torch.optim.Adam(D.parameters(),lr=getattr(cfg,'lr_d',getattr(cfg,'lr',2e-4)),betas=cfg.betas)
    schG=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_G,mode="min",patience=5,factor=.5,min_lr=1e-6);schD=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_D,mode="min",patience=5,factor=.5,min_lr=1e-6)
    best=float("inf");pat=0;history=[]
    for epoch in range(1,cfg.epochs+1):
        tr=train_one_epoch(G,D,R,train_loader,optim_G,optim_D,cfg,device);mae,rmse,vmr=validate_artificial(G,R,val_loader,cfg,device);schG.step(mae);schD.step(mae)
        row={"epoch":epoch,**tr,"V_MAE":mae,"V_RMSE":rmse,"val_miss_rate":vmr};history.append(row)
        print(f"\n[Epoch {epoch:03d}] D={tr['D']:.4f} G={tr['G']:.4f} D(real)={tr['D_real']:.3f} D(fake)={tr['D_fake']:.3f}")
        print(f"  Adv={tr['G_adv']:.4f} Rec={tr['G_rec']:.4f} Refine={tr['REFINE']:.4f} Wave={tr['WAVE']:.4f} Phy={tr['PHY']:.4f}")
        print(f"  V MAE={mae:.6f} RMSE={rmse:.6f} 人工缺失率={vmr:.4f}")
        ckpt={"model":G.state_dict(),"normalizer":norm.__dict__,"cfg":cfg.to_dict(),"epoch":epoch,"V_MAE":mae,"V_RMSE":rmse}
        if R is not None:ckpt["refiner"]=R.state_dict()
        torch.save(ckpt,out_dir/"G_last.pth")
        if mae<best:
            best=mae;pat=0;torch.save(ckpt,out_dir/"G_best.pth");print(f"  保存最佳模型，V MAE={best:.6f}")
        else:pat+=1
        if epoch%cfg.save_interval==0:torch.save(ckpt,out_dir/f"G_epoch_{epoch:03d}.pth")
        if pat>=cfg.patience:print(f"早停：epoch={epoch},best MAE={best:.6f}");break
    save_history(history,out_dir)
    try:
        fixed=fixed_ray_surrogate(G,R,cfg.val_npy,norm,cfg,device,BAD_RAYS)
        if fixed is not None:
            np.save(out_dir/"fixed_ray_V_surrogate_metrics.npy",fixed,allow_pickle=True)
            print(f"\n73/74/75固定坏射线：MAE={fixed['V_MAE_norm']:.6f},RMSE={fixed['V_RMSE_norm']:.6f},points={fixed['points']}")
    except Exception as e:print("固定坏射线评估跳过：",e)
    print("\n训练完成。");print(f"最佳模型：{out_dir/'G_best.pth'}");print(f"最终模型：{out_dir/'G_last.pth'}")

if __name__=="__main__":
    main()