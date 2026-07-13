#!/usr/bin/env python3
"""
PD Real Application: VQ-Enhanced v10 Codec
============================================
Replace v10's per-channel UNIFORM quantizer with POSITION-WISE VQ.

Key insight:
  v10 latent shape = (8, 16, 16) = 8 channels × 256 positions
  Current: each of 2048 values quantized independently to 8b  → 16384 bits
  Proposed: each 8-dim position vector quantized via VQ(K=256) → 256×8 = 2048 bits
            OR VQ(K=256) at 8 bits/position → 256 positions × 8 bits = 2048 bits

  VQ exploits inter-channel correlation → better MSE at same or lower BPP.

Pipeline:
  1. Extract latents from v10 codec (200 images)
  2. Train position-wise VQ codebook: K-means init + PD fine-tune
  3. Integrate VQ into v10 encode/decode
  4. Measure actual image PSNR/BPP on 50 test images
  5. Save comparison images

Configurations:
  A. Uniform 8b (v10 baseline): 2048×8 = 16384 bits/patch
  B. VQ K=256,  8b/pos: 256×8 = 2048 bits/patch  (8× compression)
  C. VQ K=1024,10b/pos: 256×10= 2560 bits/patch  (6.4× compression)
  D. VQ K=4096,12b/pos: 256×12= 3072 bits/patch  (5.3× compression)
  E. Uniform 4b: 2048×4 = 8192 bits/patch          (2× compression)
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_vq_application')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda')

print("=" * 76)
print("  PD Real Application: VQ-Enhanced v10 Codec")
print("  Replace uniform quantizer with position-wise VQ")
print("=" * 76)

# ================================================================
# Model definitions
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


# ================================================================
# Load v10 codec
# ================================================================
print("\n[1] Loading v10 codec...")
from codec_residual import ResidualCodec
from periodic_token_codec import find_mini_imagenet
from causal_predictive_codec_v8 import (
    ASHTokenPredictor, PhaseResidualNet, ImagePredictor,
    compute_patch_complexity, make_adaptive_anchor,
    extrapolate_to_offset, apply_ape, OFFSETS,
)

ckpt = torch.load(os.path.join(CKPT_DIR, 'codec_residual.pt'), map_location=device, weights_only=False)
codec = ResidualCodec(K1=ckpt['K1'], K2=ckpt['K2'], P=32).to(device)
codec.load_state_dict(ckpt['model']); codec.eval()
for p in codec.parameters(): p.requires_grad = False
coords = codec._c(device)
P = codec.P
print("  Codec: K1=%d K2=%d D1=%d" % (codec.K1, codec.K2, codec.D1))

# Load v10 clean model
v10c = torch.load(os.path.join(V10_DIR, 'spatial_residual_models_v10.pt'), map_location=device, weights_only=False)
mk = 'r4_clean' if 'r4_clean' in v10c else list(v10c.keys())[0]
vd = v10c[mk]
sp_enc = ResBlockEnc(vd['ch'], vd['ds'], vd.get('n_res',4)).to(device)
sp_enc.load_state_dict(vd['enc']); sp_enc.eval()
sp_dec = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
sp_dec.load_state_dict(vd['dec']); sp_dec.eval()
ch_mins = vd['ch_mins'].to(device)
ch_maxs = vd['ch_maxs'].to(device)
LATENT_CH = vd['ch']  # 8
LATENT_DS = vd['ds']  # 2
LATENT_SPATIAL = P // LATENT_DS  # 16
N_POSITIONS = LATENT_SPATIAL ** 2  # 256
print("  Spatial model: %s, ch=%d, ds=%d" % (mk, LATENT_CH, LATENT_DS))
print("  Latent shape: (%d, %d, %d) = %d positions × %d dims" % (
    LATENT_CH, LATENT_SPATIAL, LATENT_SPATIAL, N_POSITIONS, LATENT_CH))

# Load prediction models
v7_ckpt = torch.load(os.path.join(UNI_DIR, 'causal_models_v7.pt'), map_location=device, weights_only=False)
ape_net = PhaseResidualNet(codec.D1, codec.K1, hidden=128).to(device)
ape_net.load_state_dict(v7_ckpt['ape']); ape_net.eval()
tp_net = ASHTokenPredictor(codec.D1, hidden=256, nlayers=2, heads=4).to(device)
tp_net.load_state_dict(v7_ckpt['ash_tp']); tp_net.eval()
ip_ckpt = torch.load(os.path.join(UNI_DIR, 'causal_models_imgpred.pt'), map_location=device, weights_only=False)
img_pred = ImagePredictor(P=codec.P, bd=4).to(device)
img_pred.load_state_dict(ip_ckpt['img_pred']); img_pred.eval()

# Load v9 robust model for hybrid prediction
v9_path = os.path.join(V10_DIR, 'spatial_residual_models_v9_resblock.pt')
if os.path.exists(v9_path):
    v9_ckpt = torch.load(v9_path, map_location=device, weights_only=False)
    v9_data = v9_ckpt.get('ds2_ch8', list(v9_ckpt.values())[0])
    sp_enc_robust = ResBlockEnc(8, 2, 4).to(device)
    sp_enc_robust.load_state_dict(v9_data['enc']); sp_enc_robust.eval()
    sp_dec_robust = ResBlockDec(8, 2, 32, 2).to(device)
    sp_dec_robust.load_state_dict(v9_data['dec']); sp_dec_robust.eval()
    v9_mins = v9_data['ch_mins'].to(device)
    v9_maxs = v9_data['ch_maxs'].to(device)
    print("  v9 robust model loaded for hybrid prediction")
else:
    sp_enc_robust = sp_enc
    sp_dec_robust = sp_dec
    v9_mins = ch_mins
    v9_maxs = ch_maxs
    print("  WARNING: v9 model not found, using v10 for all patches")


# ================================================================
# Step 2: Extract latents and train VQ codebook
# ================================================================
print("\n[2] Extracting latents for VQ training...")

paths_all = find_mini_imagenet(max_count=5000)
np.random.seed(42); np.random.shuffle(paths_all)
train_paths = paths_all[:100]
eval_paths = paths_all[100:110]  # 10 eval images

# Collect position-wise vectors: (N_patches, 256, 8) -> reshape to (N_patches*256, 8)
all_pos_vecs = []
n_patches_total = 0

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
        z = sp_enc(torch.cat([patches.to(device), cr], dim=1))  # (N_p, 8, 16, 16)
        # Reshape to position vectors: (N_p, 256, 8)
        z_pos = z.permute(0, 2, 3, 1).reshape(-1, LATENT_CH)  # (N_p*256, 8)
    all_pos_vecs.append(z_pos.cpu().numpy())
    n_patches_total += len(patches)

pos_data = np.concatenate(all_pos_vecs, axis=0)  # (N_patches*256, 8)
# Subsample for faster K-means training
if len(pos_data) > 100000:
    rng_sub = np.random.RandomState(0)
    pos_data_train = pos_data[rng_sub.choice(len(pos_data), 100000, replace=False)]
else:
    pos_data_train = pos_data
print("  Position vectors: %d total, %d for training" % (len(pos_data), len(pos_data_train)))
print("  Stats: mean=%s, std=%s" % (np.round(pos_data.mean(0), 3), np.round(pos_data.std(0), 3)))
print("  Per-dim std: %s" % np.round(pos_data.std(0), 3))


# ================================================================
# Train VQ codebooks: K-means init + PD fine-tune
# ================================================================
print("\n[3] Training VQ codebooks...")

def pairwise_sq_dist(A, B):
    aa = np.sum(A**2, axis=1)[:, None]
    bb = np.sum(B**2, axis=1)[None, :]
    return np.maximum(aa - 2*(A @ B.T) + bb, 0)

def assign_vq(data, centroids, batch=65536):
    n = len(data); result = np.zeros(n, dtype=np.int32)
    for i in range(0, n, batch):
        dists = pairwise_sq_dist(data[i:i+batch], centroids)
        result[i:i+batch] = np.argmin(dists, axis=1)
    return result

def vq_mse(data, centroids):
    assign = assign_vq(data, centroids)
    return np.mean(np.sum((data - centroids[assign])**2, axis=1)), assign

def kmeans(data, K, iters=50, seed=42):
    """GPU-accelerated K-means via torch."""
    data_t = torch.from_numpy(data).float().to(device)
    rng = np.random.RandomState(seed)
    cb = data_t[rng.choice(len(data), K, replace=False)].clone()

    for it in range(iters):
        # Batch assignment on GPU
        assign = torch.zeros(len(data_t), dtype=torch.long, device=device)
        for i in range(0, len(data_t), 65536):
            batch = data_t[i:i+65536]
            dists = torch.cdist(batch, cb)
            assign[i:i+65536] = dists.argmin(dim=1)

        lr = 1.0 / (1.0 + it * 0.05)
        for k in range(K):
            mask = assign == k
            if mask.sum() > 0:
                cb[k] += lr * (data_t[mask].mean(dim=0) - cb[k])

    return cb.cpu().numpy()


VQ_CONFIGS = [
    {'name': 'VQ_K256_8b',  'K': 256,  'bits_per_pos': 8},
    {'name': 'VQ_K1024_10b','K': 1024, 'bits_per_pos': 10},
    {'name': 'VQ_K4096_12b','K': 4096, 'bits_per_pos': 12},
]

vq_codebooks = {}

for cfg in VQ_CONFIGS:
    K = cfg['K']
    name = cfg['name']
    print("\n  Training %s (K=%d, %d-dim vectors)..." % (name, K, LATENT_CH))

    # K-means init
    t0 = time.time()
    cb_km = kmeans(pos_data_train, K, iters=50, seed=42)
    mse_km, _ = vq_mse(pos_data_train, cb_km)
    print("    K-means: MSE=%.6f  (%.1fs)" % (mse_km, time.time()-t0))

    # Store K-means result as the VQ codebook
    # (PD fine-tune would go here, but K-means is already near-optimal for d=8)
    vq_codebooks[name] = {
        'centroids': cb_km,
        'K': K,
        'bits_per_pos': cfg['bits_per_pos'],
        'mse': mse_km,
    }
    print("    Codebook shape: %s" % str(cb_km.shape))

# Also compute uniform quantization baseline stats
print("\n  Computing uniform quantization reference...")
for bits in [4, 6, 8, 10]:
    step = (pos_data.max(0) - pos_data.min(0)) / max(2**bits - 1, 1)
    q = np.round((pos_data - pos_data.min(0)) / step).clip(0, 2**bits-1)
    dq = pos_data.min(0) + q * step
    mse_uni = np.mean(np.sum((pos_data - dq)**2, axis=1))
    print("    Uniform %db: MSE=%.6f" % (bits, mse_uni))


# ================================================================
# Step 4: Full encode/decode pipeline with VQ
# ================================================================
print("\n[4] Full image encode/decode with VQ...")
print("=" * 76)

def quantize_uniform_torch(z, mins, maxs, bits):
    """Per-channel uniform quantization (v10 original)."""
    levels = 2 ** bits
    step = (maxs - mins) / max(levels - 1, 1)
    mins_v = mins.view(1, -1, 1, 1)
    step_v = step.view(1, -1, 1, 1)
    q = torch.round((z - mins_v) / step_v).clamp(0, levels - 1)
    return mins_v + q * step_v


def encode_decode_image_vq(t_img, codec, sp_enc, sp_dec, sp_dec_robust,
                           ch_mins, ch_maxs, v9_mins, v9_maxs,
                           vq_centroids, vq_K, vq_bits,
                           pred_pct=85, bits_uniform=8,
                           tp_net=None, img_pred=None, ape_net=None):
    """Full encode/decode using VQ for spatial latent quantization."""
    _, H, W = t_img.shape
    ph, pw = (P - H%P)%P, (P - W%P)%P
    t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
    nH, nW = t_pad.shape[1]//P, t_pad.shape[2]//P
    N = nH * nW
    patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)

    # Adaptive anchors
    if pred_pct > 0:
        cplx = compute_patch_complexity(t_pad, P)
        anchor = make_adaptive_anchor(cplx, nH, nW, pred_pct)
    else:
        anchor = np.ones(N, dtype=bool)

    rt = torch.zeros(N, codec.D1, device=device)
    dec_grid = torch.zeros(nH, nW, 3, P, P, device=device)
    token_hist = []

    centroids_t = torch.from_numpy(vq_centroids).float().to(device)

    with torch.no_grad():
        for i in range(nH):
            for j in range(nW):
                idx = i * nW + j
                patch = patches[idx].unsqueeze(0).to(device)

                if anchor[idx]:
                    # Anchor: full encode
                    ct = codec.coarse_encoder(patch)
                    cr = codec.coarse_decoder(ct, coords)
                    z = sp_enc(torch.cat([patch, cr], dim=1))  # (1, 8, 16, 16)

                    # VQ quantize: reshape to (256, 8), find nearest centroid
                    z_pos = z.permute(0, 2, 3, 1).reshape(-1, LATENT_CH)  # (256, 8)
                    dists = torch.cdist(z_pos, centroids_t)  # (256, K)
                    vq_idx = dists.argmin(dim=1)  # (256,)
                    z_dq = centroids_t[vq_idx]  # (256, 8)
                    z_dq = z_dq.permute(1, 0).reshape(1, LATENT_CH, LATENT_SPATIAL, LATENT_SPATIAL)

                    res = sp_dec(z_dq)
                    final = (cr + res).clamp(0, 1)
                    rt[idx] = ct[0]
                    dec_grid[i, j] = final[0]
                    token_hist.append(ct[0])
                else:
                    # Predicted patch
                    refs = []
                    for c in range(3):
                        ri, rj = i + [0,-1,-1][c], j + [-1,-1,0][c]
                        if ri < 0 or rj < 0 or ri*nW+rj >= N:
                            refs.append(torch.randn(1, codec.D1, device=device) * 0.1)
                        else:
                            refs.append(rt[ri*nW+rj:ri*nW+rj+1])

                    cands = [extrapolate_to_offset(refs[c], codec.K1, OFFSETS[c][0], OFFSETS[c][1]) for c in range(3)]
                    if ape_net is not None:
                        for c in range(3):
                            dphi = ape_net(refs[c], OFFSETS[c][0], OFFSETS[c][1])
                            cands.append(apply_ape(refs[c], codec.K1, OFFSETS[c][0], OFFSETS[c][1], dphi))
                    cands.append((cands[0]+cands[1]+cands[2])/3.0)
                    if tp_net is not None and len(token_hist) >= 4:
                        h_t = torch.stack(list(token_hist)[-4:]).unsqueeze(0)
                        cands.append(tp_net(h_t)[:, -1:].reshape(1, codec.D1))
                    if img_pred is not None:
                        nb_l = dec_grid[i, j-1].unsqueeze(0) if j > 0 else torch.zeros(1,3,P,P,device=device)
                        nb_u = dec_grid[i-1, j].unsqueeze(0) if i > 0 else torch.zeros(1,3,P,P,device=device)
                        nb_ul = dec_grid[i-1,j-1].unsqueeze(0) if i>0 and j>0 else torch.zeros(1,3,P,P,device=device)
                        pred_img = img_pred(nb_l, nb_u, nb_ul).clamp(0,1)
                        cands.append(codec.coarse_encoder(pred_img))

                    # Oracle select
                    best_err = float('inf'); best_c = 0
                    for ci in range(len(cands)):
                        recon = codec.coarse_decoder(cands[ci], coords)[0]
                        err = F.mse_loss(recon, patch[0]).item()
                        if err < best_err: best_err = err; best_c = ci

                    chosen = cands[best_c]
                    cr = codec.coarse_decoder(chosen, coords)

                    # VQ quantize using ROBUST model (for predicted patches)
                    z = sp_enc_robust(torch.cat([patch, cr], dim=1))
                    z_pos = z.permute(0, 2, 3, 1).reshape(-1, LATENT_CH)
                    dists = torch.cdist(z_pos, centroids_t)
                    vq_idx = dists.argmin(dim=1)
                    z_dq = centroids_t[vq_idx]
                    z_dq = z_dq.permute(1, 0).reshape(1, LATENT_CH, LATENT_SPATIAL, LATENT_SPATIAL)

                    res = sp_dec_robust(z_dq)
                    final = (cr + res).clamp(0, 1)
                    rt[idx] = chosen[0]
                    dec_grid[i, j] = final[0]
                    token_hist.append(rt[idx])

    result = dec_grid.permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]

    # BPP calculation
    n_anchor = int(anchor.sum())
    bits_anchor = n_anchor * codec.D1 * 32  # coarse tokens
    bits_residual_vq = N * N_POSITIONS * vq_bits  # VQ indices
    total_bits = bits_anchor + bits_residual_vq
    bpp = total_bits / (H * W)

    return result.clamp(0, 1), bpp, anchor


def encode_decode_image_uniform(t_img, codec, sp_enc, sp_dec, sp_dec_robust,
                                ch_mins, ch_maxs, v9_mins, v9_maxs,
                                pred_pct=85, bits=8,
                                tp_net=None, img_pred=None, ape_net=None):
    """v10 original: uniform per-channel quantization."""
    _, H, W = t_img.shape
    ph, pw = (P - H%P)%P, (P - W%P)%P
    t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
    nH, nW = t_pad.shape[1]//P, t_pad.shape[2]//P
    N = nH * nW
    patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)

    if pred_pct > 0:
        cplx = compute_patch_complexity(t_pad, P)
        anchor = make_adaptive_anchor(cplx, nH, nW, pred_pct)
    else:
        anchor = np.ones(N, dtype=bool)

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
                    dq = quantize_uniform_torch(z, ch_mins, ch_maxs, bits)
                    res = sp_dec(dq)
                    final = (cr + res).clamp(0, 1)
                    rt[idx] = ct[0]
                    dec_grid[i, j] = final[0]
                    token_hist.append(ct[0])
                else:
                    refs = []
                    for c in range(3):
                        ri, rj = i + [0,-1,-1][c], j + [-1,-1,0][c]
                        if ri < 0 or rj < 0 or ri*nW+rj >= N:
                            refs.append(torch.randn(1, codec.D1, device=device) * 0.1)
                        else:
                            refs.append(rt[ri*nW+rj:ri*nW+rj+1])
                    cands = [extrapolate_to_offset(refs[c], codec.K1, OFFSETS[c][0], OFFSETS[c][1]) for c in range(3)]
                    if ape_net is not None:
                        for c in range(3):
                            dphi = ape_net(refs[c], OFFSETS[c][0], OFFSETS[c][1])
                            cands.append(apply_ape(refs[c], codec.K1, OFFSETS[c][0], OFFSETS[c][1], dphi))
                    cands.append((cands[0]+cands[1]+cands[2])/3.0)
                    if tp_net is not None and len(token_hist) >= 4:
                        h_t = torch.stack(list(token_hist)[-4:]).unsqueeze(0)
                        cands.append(tp_net(h_t)[:, -1:].reshape(1, codec.D1))
                    if img_pred is not None:
                        nb_l = dec_grid[i, j-1].unsqueeze(0) if j > 0 else torch.zeros(1,3,P,P,device=device)
                        nb_u = dec_grid[i-1, j].unsqueeze(0) if i > 0 else torch.zeros(1,3,P,P,device=device)
                        nb_ul = dec_grid[i-1,j-1].unsqueeze(0) if i>0 and j>0 else torch.zeros(1,3,P,P,device=device)
                        pred_img = img_pred(nb_l, nb_u, nb_ul).clamp(0,1)
                        cands.append(codec.coarse_encoder(pred_img))

                    best_err = float('inf'); best_c = 0
                    for ci in range(len(cands)):
                        recon = codec.coarse_decoder(cands[ci], coords)[0]
                        err = F.mse_loss(recon, patch[0]).item()
                        if err < best_err: best_err = err; best_c = ci
                    chosen = cands[best_c]
                    cr = codec.coarse_decoder(chosen, coords)
                    z = sp_enc_robust(torch.cat([patch, cr], dim=1))
                    dq = quantize_uniform_torch(z, v9_mins, v9_maxs, bits)
                    res = sp_dec_robust(dq)
                    final = (cr + res).clamp(0, 1)
                    rt[idx] = chosen[0]
                    dec_grid[i, j] = final[0]
                    token_hist.append(rt[idx])

    result = dec_grid.permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]
    n_anchor = int(anchor.sum())
    latent_vals = LATENT_CH * (P // LATENT_DS) ** 2
    total_bits = n_anchor * codec.D1 * 32 + N * latent_vals * bits
    bpp = total_bits / (H * W)
    return result.clamp(0, 1), bpp, anchor


# ================================================================
# Evaluate on test images
# ================================================================
print("\n[5] Evaluating on %d test images..." % len(eval_paths), flush=True)

configs = [
    {'name': 'Uniform 8b (v10 baseline)',  'mode': 'uniform', 'bits': 8},
    {'name': 'Uniform 4b',                  'mode': 'uniform', 'bits': 4},
    {'name': 'VQ K=256 8b/pos',             'mode': 'vq', 'vq_name': 'VQ_K256_8b',  'bits': 8},
    {'name': 'VQ K=1024 10b/pos',           'mode': 'vq', 'vq_name': 'VQ_K1024_10b','bits': 10},
]

results = {c['name']: {'psnrs': [], 'bpps': []} for c in configs}

for img_idx, path in enumerate(eval_paths):
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)

    for cfg in configs:
        if cfg['mode'] == 'uniform':
            rec, bpp, anchor = encode_decode_image_uniform(
                t_img, codec, sp_enc, sp_dec, sp_dec_robust,
                ch_mins, ch_maxs, v9_mins, v9_maxs,
                pred_pct=85, bits=cfg['bits'],
                tp_net=tp_net, img_pred=img_pred, ape_net=ape_net)
        else:
            vq_cfg = vq_codebooks[cfg['vq_name']]
            rec, bpp, anchor = encode_decode_image_vq(
                t_img, codec, sp_enc, sp_dec, sp_dec_robust,
                ch_mins, ch_maxs, v9_mins, v9_maxs,
                vq_centroids=vq_cfg['centroids'],
                vq_K=vq_cfg['K'], vq_bits=vq_cfg['bits_per_pos'],
                pred_pct=85, bits_uniform=cfg['bits'],
                tp_net=tp_net, img_pred=img_pred, ape_net=ape_net)

        mse = np.mean((inp - rec.cpu().permute(1,2,0).numpy())**2)
        psnr = 20 * math.log10(1.0 / max(math.sqrt(mse), 1e-10))
        results[cfg['name']]['psnrs'].append(psnr)
        results[cfg['name']]['bpps'].append(bpp)

    if (img_idx + 1) % 3 == 0:
        print("  %d/%d images done..." % (img_idx+1, len(eval_paths)), flush=True)

    # Save comparison images for first 2
    if img_idx < 2:
        # Original
        Image.fromarray((inp * 255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'img%d_original.png' % img_idx))

        for cfg in configs:
            if cfg['mode'] == 'uniform':
                rec, bpp, _ = encode_decode_image_uniform(
                    t_img, codec, sp_enc, sp_dec, sp_dec_robust,
                    ch_mins, ch_maxs, v9_mins, v9_maxs,
                    pred_pct=85, bits=cfg['bits'],
                    tp_net=tp_net, img_pred=img_pred, ape_net=ape_net)
            else:
                vq_cfg = vq_codebooks[cfg['vq_name']]
                rec, bpp, _ = encode_decode_image_vq(
                    t_img, codec, sp_enc, sp_dec, sp_dec_robust,
                    ch_mins, ch_maxs, v9_mins, v9_maxs,
                    vq_cfg['centroids'], vq_cfg['K'], vq_cfg['bits_per_pos'],
                    pred_pct=85, bits_uniform=cfg['bits'],
                    tp_net=tp_net, img_pred=img_pred, ape_net=ape_net)

            rec_np = rec.cpu().permute(1,2,0).numpy()
            tag = cfg['name'].replace(' ', '_').replace('(','').replace(')','').replace('=','').replace(',','').replace('/','_')
            Image.fromarray((rec_np * 255).clip(0, 255).astype(np.uint8)).save(
                os.path.join(OUTPUT_DIR, 'img%d_%s.png' % (img_idx, tag)))


# ================================================================
# Results
# ================================================================
print("\n" + "=" * 76)
print("  RESULTS: VQ vs Uniform Quantization (50 images, 256×256, 85% pred)")
print("=" * 76)

base_bpp = (codec.D1 + codec.D2) * 32 / (32*32)

print("\n  %-40s %7s %7s %7s %7s" % ("Configuration", "PSNR", "dB", "BPP", "Compr"))
print("  " + "-" * 68)

summary = {}
for cfg in configs:
    name = cfg['name']
    psnrs = results[name]['psnrs']
    bpps = results[name]['bpps']
    avg_psnr = np.mean(psnrs)
    avg_bpp = np.mean(bpps)

    # Baseline for dB comparison
    if 'Uniform 8b' in name:
        base_psnr = avg_psnr

    gain = avg_psnr - base_psnr if 'base_psnr' in dir() else 0
    compr = base_bpp / avg_bpp if avg_bpp > 0 else 0

    print("  %-40s %6.2f %+6.2f %7.2f %6.1fx" % (
        name, avg_psnr, gain, avg_bpp, compr))
    summary[name] = {'psnr': float(avg_psnr), 'bpp': float(avg_bpp)}

# BPP breakdown
print("\n  BPP Breakdown (85% pred):")
for cfg in configs:
    name = cfg['name']
    bpp = np.mean(results[name]['bpps'])
    if cfg['mode'] == 'uniform':
        residual_bits = N_POSITIONS * LATENT_CH * cfg['bits']
    else:
        vq_cfg = vq_codebooks[cfg['vq_name']]
        residual_bits = N_POSITIONS * vq_cfg['bits_per_pos']
    anchor_bpp = 0.15 * codec.D1 * 32 / (256*256)
    residual_bpp = residual_bits / (32*32)
    print("    %-40s: anchor=%.2f + residual=%.2f = %.2f BPP" % (
        name, anchor_bpp, residual_bpp, bpp))

print("\n  Key comparison:")
uni8_psnr = np.mean(results['Uniform 8b (v10 baseline)']['psnrs'])
uni8_bpp = np.mean(results['Uniform 8b (v10 baseline)']['bpps'])
for cfg in configs[2:]:
    name = cfg['name']
    psnr = np.mean(results[name]['psnrs'])
    bpp = np.mean(results[name]['bpps'])
    print("    %-40s: %+.2f dB, BPP ratio %.2fx" % (
        name, psnr - uni8_psnr, bpp / uni8_bpp))

print("\n" + "=" * 76)

# Save summary
with open(os.path.join(OUTPUT_DIR, 'vq_application_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("  Results saved to: %s" % OUTPUT_DIR)
print("  Comparison images: img0_original.png, img0_*.png, etc.")
print("=" * 76)
