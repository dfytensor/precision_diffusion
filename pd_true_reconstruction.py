#!/usr/bin/env python3
"""
PD True Reconstruction: Multi-Level Precision Recovery
=======================================================
This is the REAL Precision Diffusion reconstruction — NOT one-shot CNN decode.

Pipeline:
  Encode image → latent z₀ (10b precision)
  Forward (PD diffusion): z₀ → Q₁(z₀) → Q₂(z₀) → Q₃(z₀) → Q₄(z₀) [degrade]
  Reverse (PD recovery):  z₄ → z₃ → z₂ → z₁ → ẑ₀ [iterative refinement]

At each reverse step:
  z_{t-1} = z_t + α(t) · f_θ(z_t, t)    [predictor-guided]
  + optional Langevin noise for exploration

The predictor f_θ was trained on multi-level PD data (from validation experiments).
Step sizes α(t) are guided by Assertion 1 (correlation structure).

Comparison:
  A. Uniform 8b + CNN decode (v10 baseline, one-shot)
  B. RVQ 4-stage + CNN decode (one-shot, no PD)
  C. PD forward only (quantize at level t, decode without recovery)
  D. PD recovery (quantize at level t → iterative precision recovery → decode)
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_pd_reconstruction')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')

print("=" * 76)
print("  PD True Reconstruction: Multi-Level Precision Recovery")
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

class PDPredictor(nn.Module):
    """Predicts total residual z_0 - z_t from (z_t, t/T).
    This is the core PD model — predicts what was lost to quantization."""
    def __init__(self, d, hidden=1024, n_res=4):
        super().__init__()
        self.proj_in = nn.Linear(d + 1, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(n_res)
        ])
        self.proj_out = nn.Linear(hidden, d)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    def forward(self, z_t, t_norm):
        h = self.proj_in(torch.cat([z_t, t_norm], dim=-1))
        for b in self.blocks: h = h + b(h)
        return self.proj_out(h)

from codec_residual import ResidualCodec
from periodic_token_codec import find_mini_imagenet

# Load codec
print("\n[1] Loading v10 codec...", flush=True)
ckpt = torch.load(os.path.join(CKPT_DIR, 'codec_residual.pt'), map_location=device, weights_only=False)
codec = ResidualCodec(K1=ckpt['K1'], K2=ckpt['K2'], P=32).to(device)
codec.load_state_dict(ckpt['model']); codec.eval()
for p in codec.parameters(): p.requires_grad = False
coords = codec._c(device)
P = codec.P

v10c = torch.load(os.path.join(V10_DIR, 'spatial_residual_models_v10.pt'), map_location=device, weights_only=False)
vd = v10c['r4_clean']
sp_enc = ResBlockEnc(vd['ch'], vd['ds'], vd.get('n_res',4)).to(device)
sp_enc.load_state_dict(vd['enc']); sp_enc.eval()
for p in sp_enc.parameters(): p.requires_grad = False
sp_dec = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
sp_dec.load_state_dict(vd['dec']); sp_dec.eval()
for p in sp_dec.parameters(): p.requires_grad = False
ch_mins = vd['ch_mins'].to(device)
ch_maxs = vd['ch_maxs'].to(device)
LAT_CH, LAT_DS = vd['ch'], vd['ds']
LAT_SP = P // LAT_DS
D_LAT = LAT_CH * LAT_SP * LAT_SP  # 2048
print("  d=%d, ch=%d, ds=%d" % (D_LAT, LAT_CH, LAT_DS), flush=True)


# ================================================================
# PD quantization (forward process)
# ================================================================
T = 4
BIT_SCHEDULE = [10, 8, 6, 4, 2]
LEVELS = [2**b for b in BIT_SCHEDULE]

mins_g = None  # will set after extracting latents

def pd_quantize(z, t):
    """Quantize latent z to precision level t. z: (B, 8, 16, 16)."""
    z_flat = z.reshape(z.shape[0], -1)  # (B, 2048)
    nl = LEVELS[t]
    step = (maxs_g - mins_g) / max(nl - 1, 1)
    q = torch.round((z_flat - mins_g) / step).clamp(0, nl - 1)
    return (mins_g + q * step).reshape_as(z)


# ================================================================
# Extract latents and train PD predictor
# ================================================================
print("\n[2] Extracting latents...", flush=True)
paths = find_mini_imagenet(max_count=5000)
np.random.seed(42); np.random.shuffle(paths)
train_paths = paths[:80]
eval_paths = paths[80:95]

all_z = []; all_cr = []; all_patches = []
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
    all_z.append(z); all_cr.append(cr); all_patches.append(patches)

z_train = torch.cat(all_z)  # (N, 8, 16, 16)
cr_train = torch.cat(all_cr)
patch_train = torch.cat(all_patches)
print("  %d training patches" % len(z_train), flush=True)

# Set global quantization range
z_flat_train = z_train.reshape(len(z_train), -1)
mins_g = z_flat_train.min(dim=0)[0]
maxs_g = z_flat_train.max(dim=0)[0]
print("  Latent range: [%.3f, %.3f]" % (mins_g.min().item(), maxs_g.max().item()), flush=True)

# Precompute quantized versions at each level
print("\n[3] Precomputing PD quantization levels...", flush=True)
z_q_levels = {}
for t in range(T + 1):
    z_q_levels[t] = pd_quantize(z_train, t).detach()
    mse = F.mse_loss(z_q_levels[t], z_train).item()
    print("  t=%d (%db, %d levels): MSE=%.6f" % (t, BIT_SCHEDULE[t], LEVELS[t], mse), flush=True)


# ================================================================
# Train PD predictor: f(z_t, t) → z_0 - z_t
# ================================================================
print("\n[4] Training PD predictor f(z_t, t) → z_0 - z_t...", flush=True)

predictor = PDPredictor(D_LAT, hidden=1024, n_res=4).to(device)
n_params = sum(p.numel() for p in predictor.parameters())
print("  Predictor: %d params (%.1fM)" % (n_params, n_params / 1e6), flush=True)

opt = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)
N = len(z_train)
bs = 512

for ep in range(200):
    perm = torch.randperm(N)
    total_loss = 0; nb = 0
    for i in range(0, N, bs):
        idx = perm[i:i+bs]
        z0 = z_train[idx]
        # Randomly sample precision level t ∈ {1,...,T}
        t_val = np.random.randint(1, T + 1)
        z_t = z_q_levels[t_val][idx]
        target = z0 - z_t  # total quantization residual

        z_t_flat = z_t.reshape(len(idx), -1).detach()
        target_flat = target.reshape(len(idx), -1).detach()
        tn = torch.full((len(idx), 1), t_val / T, device=device)

        pred = predictor(z_t_flat, tn)
        loss = F.mse_loss(pred, target_flat)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); nb += 1
    sched.step()
    if (ep + 1) % 50 == 0 or ep == 0:
        # Evaluate predictor quality at each level
        with torch.no_grad():
            eval_stats = []
            for t_val in range(1, T + 1):
                z_t_flat = z_q_levels[t_val][:1000].reshape(1000, -1)
                tn = torch.full((1000, 1), t_val / T, device=device)
                pred = predictor(z_t_flat, tn)
                target = (z_train[:1000] - z_q_levels[t_val][:1000]).reshape(1000, -1)
                pred_mse = F.mse_loss(pred, target).item()
                direct_mse = F.mse_loss(torch.zeros_like(target), target).item()
                ratio = pred_mse / max(direct_mse, 1e-10)
                eval_stats.append("t%d: %.4f/%.4f=%.2f" % (t_val, pred_mse, direct_mse, ratio))
        print("  Epoch %3d: loss=%.6f | %s" % (ep+1, total_loss/nb, "  ".join(eval_stats)), flush=True)

predictor.eval()
print("  Done.", flush=True)


# ================================================================
# PD Reconstruction: iterative precision recovery
# ================================================================
print("\n[5] PD Reconstruction: iterative precision recovery", flush=True)

def pd_reconstruct(z_coarse, t_start, predictor, schedule, langevin_scale=0.0):
    """Recover precision from t_start to t=0.
    
    z_coarse: (B, 8, 16, 16) quantized at t_start
    schedule: list of (t_target, step_size, n_iters)
    
    Returns: recovered latent (B, 8, 16, 16)
    """
    z = z_coarse.clone()
    B = z.shape[0]
    
    for t_target, alpha, n_iters in schedule:
        t_norm = torch.full((B, 1), t_target / T, device=device)
        for it in range(n_iters):
            z_flat = z.reshape(B, -1)
            pred_residual = predictor(z_flat, t_norm)  # (B, 2048)
            pred_residual = pred_residual.reshape_as(z)
            
            # Small step in predicted direction
            z = z + alpha * pred_residual
            
            # Optional Langevin noise for exploration
            if langevin_scale > 0:
                noise = torch.randn_like(z) * langevin_scale / math.sqrt(D_LAT)
                z = z + noise
            
            # Clamp to valid range
            z = z.clamp(mins_g.min(), maxs_g.max())
    
    return z


# Define recovery schedules based on Assertion 1 guidance
# Focus effort on coarse levels (high correlation), less on fine levels
schedules = {
    'PD_t4_full': [
        # (target_level, step_size, n_iters)
        # Based on corr: t4=0.099 (strong), t3=0.064 (medium), t2=0.026 (weak), t1=0.024 (weak)
        (4, 0.3, 5),   # t=4: strong structure, bold steps
        (3, 0.2, 5),   # t=3: medium structure
        (2, 0.1, 3),   # t=2: weak structure, conservative
        (1, 0.05, 2),  # t=1: nearly random, minimal
    ],
    'PD_t4_conservative': [
        (4, 0.15, 8),
        (3, 0.10, 8),
        (2, 0.05, 5),
        (1, 0.02, 3),
    ],
    'PD_t4_aggressive': [
        (4, 0.5, 3),
        (3, 0.3, 3),
        (2, 0.15, 2),
        (1, 0.1, 1),
    ],
    'PD_t2_only': [
        # Only recover from t=2 (6b) — less aggressive
        (2, 0.2, 5),
        (1, 0.1, 3),
    ],
    'PD_t4_langevin': [
        # With Langevin noise for exploration
        (4, 0.3, 5),
        (3, 0.2, 5),
        (2, 0.1, 3),
        (1, 0.05, 2),
    ],
}


# ================================================================
# Evaluate all configurations
# ================================================================
print("\n[6] Evaluating reconstruction quality...", flush=True)

def reconstruct_image(t_img, codec, sp_enc, sp_dec, coords, mode='baseline_uniform8b', **kwargs):
    """Full image reconstruction. Returns (result_tensor, bpp)."""
    _, H, W = t_img.shape
    ph, pw = (P-H%P)%P, (P-W%P)%P
    t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
    nH, nW = t_pad.shape[1]//P, t_pad.shape[2]//P
    N = nH * nW
    patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)

    dec_grid = torch.zeros(nH, nW, 3, P, P, device=device)

    with torch.no_grad():
        patches_gpu = patches.to(device)
        ct = codec.coarse_encoder(patches_gpu)
        cr = codec.coarse_decoder(ct, coords)
        z0 = sp_enc(torch.cat([patches_gpu, cr], dim=1))  # original latent

        if mode == 'baseline_uniform8b':
            # Standard v10: uniform 8b quantize + CNN decode
            z_dq = pd_quantize(z0, 1)  # t=1 = 8b
            res = sp_dec(z_dq)
            bits_per_pos = 8 * LAT_CH
        elif mode == 'no_recovery_t4':
            # Quantize at t=4 (2b), decode WITHOUT recovery — worst case
            z_dq = pd_quantize(z0, 4)
            res = sp_dec(z_dq)
            bits_per_pos = 2 * LAT_CH
        elif mode == 'no_recovery_t2':
            # Quantize at t=2 (6b), decode WITHOUT recovery
            z_dq = pd_quantize(z0, 2)
            res = sp_dec(z_dq)
            bits_per_pos = 6 * LAT_CH
        elif mode.startswith('PD_'):
            # PD recovery: quantize at coarse level, then recover precision
            schedule_name = mode
            schedule = schedules[schedule_name]
            t_start = schedule[0][0]
            z_coarse = pd_quantize(z0, t_start)

            # Determine langevin
            lv = kwargs.get('langevin', 0.0)
            if schedule_name == 'PD_t4_langevin':
                lv = 0.5

            z_recovered = pd_reconstruct(z_coarse, t_start, predictor, schedule, langevin_scale=lv)
            res = sp_dec(z_recovered)

            # BPP: only need to store the coarse-level quantization indices
            bits_per_pos = BIT_SCHEDULE[t_start] * LAT_CH
        else:
            raise ValueError("Unknown mode: %s" % mode)

        finals = (cr + res).clamp(0, 1)
        for idx in range(N):
            i2, j2 = idx // nW, idx % nW
            dec_grid[i2, j2] = finals[idx]

    result = dec_grid.permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]
    total_bits = N * codec.D1 * 32 + N * (LAT_SP**2) * bits_per_pos
    bpp = total_bits / (H * W)
    return result.clamp(0, 1), bpp


# Test configurations
test_configs = [
    ('baseline_uniform8b', 'Uniform 8b (v10 baseline, one-shot)'),
    ('no_recovery_t2', 'No recovery: quantize@6b (one-shot)'),
    ('no_recovery_t4', 'No recovery: quantize@2b (one-shot)'),
    ('PD_t2_only', 'PD recovery from 6b (t=2→0)'),
    ('PD_t4_full', 'PD recovery from 2b (t=4→0, full)'),
    ('PD_t4_conservative', 'PD recovery from 2b (conservative)'),
    ('PD_t4_aggressive', 'PD recovery from 2b (aggressive)'),
    ('PD_t4_langevin', 'PD recovery from 2b (+ Langevin)'),
]

results = {}
for img_idx, path in enumerate(eval_paths):
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)

    for mode, label in test_configs:
        rec, bpp = reconstruct_image(t_img, codec, sp_enc, sp_dec, coords, mode=mode)
        mse = np.mean((inp - rec.cpu().permute(1,2,0).numpy())**2)
        psnr = 20 * math.log10(1.0 / max(math.sqrt(mse), 1e-10))
        if mode not in results:
            results[mode] = {'label': label, 'psnrs': [], 'bpps': []}
        results[mode]['psnrs'].append(psnr)
        results[mode]['bpps'].append(bpp)

    # Save comparison images for first 3
    if img_idx < 3:
        Image.fromarray((inp*255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'rec%d_00original.png' % img_idx))
        for si, (mode, label) in enumerate(test_configs):
            rec, bpp = reconstruct_image(t_img, codec, sp_enc, sp_dec, coords, mode=mode)
            tag = mode.replace('_','')
            psnr_val = results[mode]['psnrs'][img_idx]
            Image.fromarray((rec.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)).save(
                os.path.join(OUTPUT_DIR, 'rec%d_%02d%s_%.1fdB.png' % (img_idx, si+1, tag, psnr_val)))

    print("  %d/%d done" % (img_idx+1, len(eval_paths)), flush=True)


# ================================================================
# Results
# ================================================================
print("\n" + "=" * 76)
print("  PD RECONSTRUCTION RESULTS (%d images, 0%% pred)" % len(eval_paths))
print("=" * 76)

print("\n  %-45s %7s %7s %7s" % ("Configuration", "PSNR", "BPP", "dB vs U8"))
print("  " + "-" * 68)

u8_psnr = results.get('baseline_uniform8b', {}).get('psnrs', [0])
u8_avg = np.mean(u8_psnr) if u8_psnr else 0

summary = {}
for mode, label in test_configs:
    if mode not in results: continue
    r = results[mode]
    psnr = np.mean(r['psnrs'])
    bpp = np.mean(r['bpps'])
    diff = psnr - u8_avg
    print("  %-45s %6.2f %7.2f %+7.2f" % (label, psnr, bpp, diff))
    summary[mode] = {'label': label, 'psnr': float(psnr), 'bpp': float(bpp)}

# Latent-level recovery analysis
print("\n  Latent-level PD Recovery Analysis:")
print("  (How much precision does the predictor recover?)")
with torch.no_grad():
    for t_start in [4, 2]:
        z_orig = z_train[:500]
        z_coarse = pd_quantize(z_orig, t_start)
        mse_before = F.mse_loss(z_coarse, z_orig).item()

        # Apply PD recovery
        schedule = schedules['PD_t4_full'] if t_start == 4 else schedules['PD_t2_only']
        z_recovered = pd_reconstruct(z_coarse, t_start, predictor, schedule)
        mse_after = F.mse_loss(z_recovered, z_orig).item()

        recovery_pct = (1 - mse_after / mse_mse) * 100 if (mse_mse := mse_before) > 0 else 0
        print("    t=%d (%db→10b): before=%.6f → after=%.6f (%.1f%% recovery)" % (
            t_start, BIT_SCHEDULE[t_start], mse_before, mse_after,
            (1 - mse_after/mse_before) * 100 if mse_before > 0 else 0))

print("\n" + "=" * 76)

# Save
with open(os.path.join(OUTPUT_DIR, 'pd_reconstruction.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("  Results: %s" % OUTPUT_DIR)
print("  Images: rec0_00original.png, rec0_04PDt2only_*.png, etc.")
print("=" * 76)
