#!/usr/bin/env python3
"""
PD Real Application v2: Residual VQ (RVQ) for v10 Codec
=========================================================
Multi-stage VQ = PD's multi-level forward process applied to compression.

Problem with v1: position-wise VQ K=256 gives 8 bits/position,
while uniform 8b gives 64 bits/position (8x more). Unfair comparison.

Solution: Residual VQ (RVQ) — multiple VQ stages on residuals:
  Stage 1: VQ K=256 on latent → coarse (8 bits)
  Stage 2: VQ K=256 on residual → refine (8 bits)
  Stage 3: VQ K=256 on residual → refine (8 bits)
  Stage 4: VQ K=256 on residual → refine (8 bits)
  Total: 32 bits/position = 8192 bits/patch (vs uniform 8b: 16384)

Each stage is a PD diffusion step. Fine-tune decoder for RVQ input.

Key test: Can RVQ at 32 bits match uniform 8b at 64 bits per position?
"""

import sys, os, math, time, json
sys.path.insert(0, 'F:\\tmp_pytorch')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

UNI_DIR = 'F:\\OpenASH\\vision_voc\\uniencode'
V10_DIR = os.path.join(UNI_DIR, 'v10_bundle')
CKPT_DIR = os.path.join(UNI_DIR, 'periodic_codec_checkpoints')
sys.path.insert(0, V10_DIR)
sys.path.insert(0, UNI_DIR)

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_rvq_application')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')

print("=" * 76)
print("  PD Application v2: Residual VQ (Multi-Stage VQ)")
print("  Multi-level quantization = PD forward process")
print("=" * 76, flush=True)

# ================================================================
# Models
# ================================================================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(ch,ch,3,padding=1), nn.GELU(), nn.Conv2d(ch,ch,3,padding=1))
        self.a = nn.Parameter(torch.tensor(0.1))
    def forward(self, x): return x + self.a * self.block(x)

class ResBlockEnc(nn.Module):
    def __init__(self, ch=8, ds=2, n_res=4, hidden=128):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(6, hidden, 3, padding=1), nn.GELU())
        self.res = nn.Sequential(*[ResBlock(hidden) for _ in range(n_res)])
        self.post = nn.Conv2d(hidden, ch, 3, stride=ds, padding=1) if ds > 1 else nn.Conv2d(hidden, ch, 3, padding=1)
        self.ds, self.ch = ds, ch
    def forward(self, x): return self.post(self.res(self.pre(x)))

class ResBlockDec(nn.Module):
    def __init__(self, ch=8, ds=2, ts=32, n_res=2, hidden=128):
        super().__init__()
        self.ts, self.ds, self.ch = ts, ds, ch
        self.pre = nn.Sequential(nn.Conv2d(ch, hidden, 3, padding=1), nn.GELU())
        self.res = nn.Sequential(*[ResBlock(hidden) for _ in range(n_res)])
        self.post = nn.Conv2d(hidden, 3, 3, padding=1)
    def forward(self, z):
        if self.ds > 1:
            z = F.interpolate(z, size=(self.ts, self.ts), mode='bilinear', align_corners=False)
        return self.post(self.res(self.pre(z)))

from codec_residual import ResidualCodec
from periodic_token_codec import find_mini_imagenet
from causal_predictive_codec_v8 import (
    ASHTokenPredictor, PhaseResidualNet, ImagePredictor,
    compute_patch_complexity, make_adaptive_anchor,
    extrapolate_to_offset, apply_ape, OFFSETS,
)

# Load codec
print("\n[1] Loading v10 codec...", flush=True)
ckpt = torch.load(os.path.join(CKPT_DIR, 'codec_residual.pt'), map_location=device, weights_only=False)
codec = ResidualCodec(K1=ckpt['K1'], K2=ckpt['K2'], P=32).to(device)
codec.load_state_dict(ckpt['model']); codec.eval()
for p in codec.parameters(): p.requires_grad = False
coords = codec._c(device)
P = codec.P

v10c = torch.load(os.path.join(V10_DIR, 'spatial_residual_models_v10.pt'), map_location=device, weights_only=False)
mk = 'r4_clean' if 'r4_clean' in v10c else list(v10c.keys())[0]
vd = v10c[mk]
sp_enc = ResBlockEnc(vd['ch'], vd['ds'], vd.get('n_res',4)).to(device)
sp_enc.load_state_dict(vd['enc']); sp_enc.eval()
sp_dec = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
sp_dec.load_state_dict(vd['dec']); sp_dec.eval()
ch_mins = vd['ch_mins'].to(device)
ch_maxs = vd['ch_maxs'].to(device)
LAT_CH = vd['ch']; LAT_DS = vd['ds']; LAT_SP = P // LAT_DS
N_POS = LAT_SP ** 2
print("  Loaded: ch=%d ds=%d positions=%d" % (LAT_CH, LAT_DS, N_POS), flush=True)

# ================================================================
# Extract latents for RVQ training
# ================================================================
print("\n[2] Extracting latents...", flush=True)
paths = find_mini_imagenet(max_count=5000)
np.random.seed(42); np.random.shuffle(paths)
train_paths = paths[:80]

all_z = []
all_cr = []
all_patches = []
for path in train_paths:
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)
    H, W = t_img.shape[1], t_img.shape[2]
    ph, pw = (P-H%P)%P, (P-W%P)%P
    t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
    patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)
    with torch.no_grad():
        ct = codec.coarse_encoder(patches.to(device))
        cr = codec.coarse_decoder(ct, coords)
        z = sp_enc(torch.cat([patches.to(device), cr], dim=1))
    all_z.append(z.cpu())
    all_cr.append(cr.cpu())
    all_patches.append(patches)

z_all = torch.cat(all_z, dim=0)  # (N_patches, 8, 16, 16)
cr_all = torch.cat(all_cr, dim=0)
patch_all = torch.cat(all_patches, dim=0)
print("  %d patches collected" % len(z_all), flush=True)

# Reshape to position vectors: (N_patches * 256, 8)
pos_vecs = z_all.permute(0, 2, 3, 1).reshape(-1, LAT_CH).numpy()
print("  Position vectors: %d x %d" % pos_vecs.shape, flush=True)

# Subsample
rng_s = np.random.RandomState(0)
pos_train = pos_vecs[rng_s.choice(len(pos_vecs), min(50000, len(pos_vecs)), replace=False)]


# ================================================================
# Train multi-stage RVQ codebooks
# ================================================================
print("\n[3] Training RVQ codebooks (multi-stage)...", flush=True)

def kmeans_gpu(data_np, K, iters=50, seed=42):
    data_t = torch.from_numpy(data_np).float().to(device)
    rng = np.random.RandomState(seed)
    cb = data_t[rng.choice(len(data_t), K, replace=False)].clone()
    for it in range(iters):
        assign = torch.zeros(len(data_t), dtype=torch.long, device=device)
        for i in range(0, len(data_t), 65536):
            dists = torch.cdist(data_t[i:i+65536], cb)
            assign[i:i+65536] = dists.argmin(dim=1)
        lr = 1.0 / (1.0 + it * 0.05)
        for k in range(K):
            mask = assign == k
            if mask.sum() > 0:
                cb[k] += lr * (data_t[mask].mean(dim=0) - cb[k])
    return cb

N_STAGES = 4
K_PER_STAGE = 256
BITS_PER_STAGE = 8  # log2(256)

rvq_codebooks = []
residual = pos_train.copy()

for stage in range(N_STAGES):
    print("  Stage %d: K=%d, residual MSE=%.6f" % (stage, K_PER_STAGE, np.mean(residual**2)), flush=True)
    cb = kmeans_gpu(residual, K_PER_STAGE, iters=40, seed=42+stage)
    cb_np = cb.cpu().numpy()

    # Quantize and compute residual
    data_t = torch.from_numpy(residual).float().to(device)
    with torch.no_grad():
        dists = torch.cdist(data_t, cb)
        assign = dists.argmin(dim=1)
        quantized = cb[assign].cpu().numpy()

    residual = residual - quantized
    rvq_codebooks.append(cb_np)
    mse_after = np.mean(residual**2)
    print("    After stage %d: residual MSE=%.6f (cumulative bits=%d)" % (
        stage, mse_after, (stage+1)*BITS_PER_STAGE), flush=True)

# Compare RVQ vs uniform at same bit budget
print("\n  RVQ vs Uniform comparison (latent-level MSE):", flush=True)

# Compute RVQ reconstruction properly
rvq_recon_full = np.zeros_like(pos_train)
res = pos_train.copy()
rvq_mses = []
for s in range(N_STAGES):
    cb = rvq_codebooks[s]
    data_t = torch.from_numpy(res).float().to(device)
    cb_t = torch.from_numpy(cb).float().to(device)
    with torch.no_grad():
        dists = torch.cdist(data_t, cb_t)
        assign = dists.argmin(dim=1)
        q = cb[assign.cpu().numpy()]
    rvq_recon_full += q
    res = res - q
    mse = np.mean((pos_train - rvq_recon_full)**2)
    rvq_mses.append(mse)
    print("    RVQ %d stages (%d bits/pos): MSE=%.6f" % (s+1, (s+1)*8, mse), flush=True)

# Uniform comparison
print("", flush=True)
mins = pos_train.min(0)
maxs = pos_train.max(0)
for bits in [8, 16, 24, 32, 40, 48, 64]:
    step = (maxs - mins) / max(2**bits - 1, 1)
    q = np.round((pos_train - mins) / step).clip(0, 2**bits - 1)
    dq = mins + q * step
    mse = np.mean((pos_train - dq)**2)
    bits_per_pos = bits
    print("    Uniform %d bits/value (%d bits/pos): MSE=%.6f" % (bits, bits*8, mse), flush=True)


# ================================================================
# Full pipeline: RVQ encode/decode on actual images
# ================================================================
print("\n[4] Full image encode/decode with RVQ...", flush=True)

def rvq_encode(z_latent, codebooks):
    """Encode latent using multi-stage RVQ. Returns indices per stage."""
    # z_latent: (1, 8, 16, 16)
    z_pos = z_latent.permute(0, 2, 3, 1).reshape(-1, LAT_CH)  # (256, 8)
    residual = z_pos.clone()
    indices = []
    for cb in codebooks:
        cb_t = torch.from_numpy(cb).float().to(z_latent.device)
        dists = torch.cdist(residual, cb_t)
        idx = dists.argmin(dim=1)
        indices.append(idx)
        residual = residual - cb_t[idx]
    return indices

def rvq_decode(indices, codebooks, device):
    """Decode RVQ indices back to latent."""
    z_pos = torch.zeros(N_POS, LAT_CH, device=device)
    for s, idx in enumerate(indices):
        cb_t = torch.from_numpy(codebooks[s]).float().to(device)
        z_pos += cb_t[idx]
    return z_pos.permute(1, 0).reshape(1, LAT_CH, LAT_SP, LAT_SP)

def quantize_uniform(z, mins, maxs, bits):
    levels = 2 ** bits
    step = (maxs - mins) / max(levels - 1, 1)
    mins_v = mins.view(1, -1, 1, 1)
    step_v = step.view(1, -1, 1, 1)
    q = torch.round((z - mins_v) / step_v).clamp(0, levels - 1)
    return mins_v + q * step_v


eval_paths = paths[80:90]  # 10 eval images

print("  Evaluating %d images with %d configurations..." % (len(eval_paths), N_STAGES+1), flush=True)

configs = []
# RVQ with 1,2,3,4 stages
for n_s in range(1, N_STAGES+1):
    configs.append({
        'name': 'RVQ %d stages (%d bits/pos)' % (n_s, n_s * BITS_PER_STAGE),
        'mode': 'rvq', 'n_stages': n_s,
        'bits_per_pos': n_s * BITS_PER_STAGE,
    })
# Uniform baselines at comparable bits
for bits in [4, 6, 8]:
    configs.append({
        'name': 'Uniform %db (%d bits/pos)' % (bits, bits * LAT_CH),
        'mode': 'uniform', 'bits': bits,
        'bits_per_pos': bits * LAT_CH,
    })

results = {c['name']: {'psnrs': [], 'bpps': []} for c in configs}

for img_idx, path in enumerate(eval_paths):
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)

    _, H, W = t_img.shape
    ph, pw = (P-H%P)%P, (P-W%P)%P
    t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
    nH, nW = t_pad.shape[1]//P, t_pad.shape[2]//P
    N = nH * nW
    patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)

    cplx = compute_patch_complexity(t_pad, P)
    anchor = make_adaptive_anchor(cplx, nH, nW, 85)
    n_anchor = int(anchor.sum())

    for cfg in configs:
        rt = torch.zeros(N, codec.D1, device=device)
        dec_grid = torch.zeros(nH, nW, 3, P, P, device=device)
        token_hist = []

        with torch.no_grad():
            for i in range(nH):
                for j in range(nW):
                    idx = i * nW + j
                    patch = patches[idx].unsqueeze(0).to(device)

                    if anchor[idx]:
                        ct = codec.coarse_encoder(patch)
                        cr = codec.coarse_decoder(ct, coords)
                        z = sp_enc(torch.cat([patch, cr], dim=1))

                        if cfg['mode'] == 'rvq':
                            indices = rvq_encode(z, rvq_codebooks[:cfg['n_stages']])
                            z_dq = rvq_decode(indices, rvq_codebooks[:cfg['n_stages']], device)
                        else:
                            z_dq = quantize_uniform(z, ch_mins, ch_maxs, cfg['bits'])

                        res = sp_dec(z_dq)
                        final = (cr + res).clamp(0, 1)
                        rt[idx] = ct[0]
                        dec_grid[i, j] = final[0]
                        token_hist.append(ct[0])
                    else:
                        # Predicted patch - simplified (periodic extrapolation only)
                        refs = []
                        for c in range(3):
                            ri, rj = i + [0,-1,-1][c], j + [-1,-1,0][c]
                            if ri < 0 or rj < 0 or ri*nW+rj >= N:
                                refs.append(torch.randn(1, codec.D1, device=device) * 0.1)
                            else:
                                refs.append(rt[ri*nW+rj:ri*nW+rj+1])
                        cands = [extrapolate_to_offset(refs[c], codec.K1, OFFSETS[c][0], OFFSETS[c][1]) for c in range(3)]
                        cands.append((cands[0]+cands[1]+cands[2])/3.0)

                        best_err = float('inf'); best_c = 0
                        for ci in range(len(cands)):
                            recon = codec.coarse_decoder(cands[ci], coords)[0]
                            err = F.mse_loss(recon, patch[0]).item()
                            if err < best_err: best_err = err; best_c = ci
                        chosen = cands[best_c]
                        cr = codec.coarse_decoder(chosen, coords)
                        z = sp_enc(torch.cat([patch, cr], dim=1))

                        if cfg['mode'] == 'rvq':
                            indices = rvq_encode(z, rvq_codebooks[:cfg['n_stages']])
                            z_dq = rvq_decode(indices, rvq_codebooks[:cfg['n_stages']], device)
                        else:
                            z_dq = quantize_uniform(z, ch_mins, ch_maxs, cfg['bits'])

                        res = sp_dec(z_dq)
                        final = (cr + res).clamp(0, 1)
                        rt[idx] = chosen[0]
                        dec_grid[i, j] = final[0]
                        token_hist.append(rt[idx])

        result = dec_grid.permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]
        mse = np.mean((inp - result.cpu().permute(1,2,0).numpy())**2)
        psnr = 20 * math.log10(1.0 / max(math.sqrt(mse), 1e-10))
        results[cfg['name']]['psnrs'].append(psnr)

        bits_per_patch = N_POS * cfg['bits_per_pos']
        total_bits = n_anchor * codec.D1 * 32 + N * bits_per_patch
        bpp = total_bits / (H * W)
        results[cfg['name']]['bpps'].append(bpp)

    # Save comparison for first 2 images
    if img_idx < 2:
        Image.fromarray((inp * 255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'img%d_orig.png' % img_idx))
        for cfg in configs:
            # Reconstruct with this config
            rt2 = torch.zeros(N, codec.D1, device=device)
            dg2 = torch.zeros(nH, nW, 3, P, P, device=device)
            th2 = []
            with torch.no_grad():
                for i in range(nH):
                    for j in range(nW):
                        idx2 = i*nW+j
                        patch2 = patches[idx2].unsqueeze(0).to(device)
                        if anchor[idx2]:
                            ct2 = codec.coarse_encoder(patch2)
                            cr2 = codec.coarse_decoder(ct2, coords)
                            z2 = sp_enc(torch.cat([patch2, cr2], dim=1))
                            if cfg['mode'] == 'rvq':
                                idx_r = rvq_encode(z2, rvq_codebooks[:cfg['n_stages']])
                                z_dq2 = rvq_decode(idx_r, rvq_codebooks[:cfg['n_stages']], device)
                            else:
                                z_dq2 = quantize_uniform(z2, ch_mins, ch_maxs, cfg['bits'])
                            res2 = sp_dec(z_dq2)
                            dg2[i,j] = (cr2+res2).clamp(0,1)[0]
                            rt2[idx2] = ct2[0]; th2.append(ct2[0])
                        else:
                            refs2 = []
                            for c in range(3):
                                ri2,rj2 = i+[0,-1,-1][c], j+[-1,-1,0][c]
                                if ri2<0 or rj2<0 or ri2*nW+rj2>=N:
                                    refs2.append(torch.randn(1,codec.D1,device=device)*0.1)
                                else:
                                    refs2.append(rt2[ri2*nW+rj2:ri2*nW+rj2+1])
                            cs2 = [extrapolate_to_offset(refs2[c],codec.K1,OFFSETS[c][0],OFFSETS[c][1]) for c in range(3)]
                            cs2.append((cs2[0]+cs2[1]+cs2[2])/3)
                            be2 = float('inf'); bc2 = 0
                            for ci2 in range(len(cs2)):
                                r2 = codec.coarse_decoder(cs2[ci2],coords)[0]
                                e2 = F.mse_loss(r2,patch2[0]).item()
                                if e2<be2: be2=e2; bc2=ci2
                            ch2 = cs2[bc2]
                            cr2 = codec.coarse_decoder(ch2,coords)
                            z2 = sp_enc(torch.cat([patch2,cr2],dim=1))
                            if cfg['mode']=='rvq':
                                idx_r = rvq_encode(z2, rvq_codebooks[:cfg['n_stages']])
                                z_dq2 = rvq_decode(idx_r, rvq_codebooks[:cfg['n_stages']], device)
                            else:
                                z_dq2 = quantize_uniform(z2, ch_mins, ch_maxs, cfg['bits'])
                            res2 = sp_dec(z_dq2)
                            dg2[i,j] = (cr2+res2).clamp(0,1)[0]
                            rt2[idx2] = ch2[0]; th2.append(rt2[idx2])
            rec2 = dg2.permute(2,0,3,1,4).reshape(3,nH*P,nW*P)[:,:H,:W]
            tag = cfg['name'].split('(')[0].strip().replace(' ','_')
            Image.fromarray((rec2.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)).save(
                os.path.join(OUTPUT_DIR, 'img%d_%s.png' % (img_idx, tag)))

    print("  %d/%d done" % (img_idx+1, len(eval_paths)), flush=True)

# ================================================================
# Results
# ================================================================
print("\n" + "=" * 76)
print("  RESULTS: RVQ vs Uniform (10 images, 256×256, 85% pred)")
print("=" * 76)

print("\n  %-40s %7s %7s %8s %7s" % ("Config", "PSNR", "BPP", "bits/pos", "Compr"))
print("  " + "-" * 70)

summary = {}
for cfg in configs:
    name = cfg['name']
    psnr = np.mean(results[name]['psnrs'])
    bpp = np.mean(results[name]['bpps'])
    base_bpp = (codec.D1 + codec.D2) * 32 / (32*32)
    compr = base_bpp / bpp if bpp > 0 else 0
    print("  %-40s %6.2f %7.2f %8d %6.1fx" % (
        name, psnr, bpp, cfg['bits_per_pos'], compr))
    summary[name] = {'psnr': float(psnr), 'bpp': float(bpp), 'bits_per_pos': cfg['bits_per_pos']}

# Rate-quality analysis
print("\n  Rate-Quality Analysis:")
print("  At similar BPP, which gives better PSNR?")
# Group by approximate BPP ranges
for target_bpp in [6, 10, 14, 18]:
    print("\n    BPP ~ %d:" % target_bpp)
    for cfg in configs:
        name = cfg['name']
        bpp = np.mean(results[name]['bpps'])
        if abs(bpp - target_bpp) < 4:
            psnr = np.mean(results[name]['psnrs'])
            print("      %-40s PSNR=%.2f @ %.2f BPP" % (name, psnr, bpp))

print("\n" + "=" * 76)
with open(os.path.join(OUTPUT_DIR, 'rvq_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("  Results: %s" % OUTPUT_DIR)
print("  Images: img0_orig.png, img0_RVQ_*.png, img0_Uniform_*.png")
print("=" * 76)
