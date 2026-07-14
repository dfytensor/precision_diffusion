#!/usr/bin/env python3
"""
PD Reconstruction v2: + Decoder Fine-tuning
=============================================
v1 showed PD predictor recovers 80% precision at t=4 (2b→10b),
but image PSNR only 24.98 dB due to decoder mismatch.

v2: Fine-tune ResBlockDec for PD-recovered latents.
The decoder learns to handle the residual quantization artifacts
that the predictor couldn't fully remove.

Pipeline:
  1. Train PD predictor (done in v1)
  2. Generate PD-recovered training data:
     z₀ → quantize@t → PD recover → ẑ₀
  3. Fine-tune decoder: Dec(ẑ₀) → residual image
  4. Evaluate on test images

Also test: PD recovery at t=4 (2b) + t=2 (6b) with fine-tuned decoder.
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_pd_reconstruction_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')

print("=" * 76)
print("  PD Reconstruction v2: + Decoder Fine-tuning")
print("=" * 76, flush=True)

# ================================================================
# Models (same as v1)
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
ch_mins = vd['ch_mins'].to(device)
ch_maxs = vd['ch_maxs'].to(device)
LAT_CH, LAT_DS = vd['ch'], vd['ds']
LAT_SP = P // LAT_DS
D_LAT = LAT_CH * LAT_SP * LAT_SP

sp_dec_orig = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
sp_dec_orig.load_state_dict(vd['dec']); sp_dec_orig.eval()
for p in sp_dec_orig.parameters(): p.requires_grad = False
print("  Loaded. d=%d" % D_LAT, flush=True)


# ================================================================
# Extract latents
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

z_train = torch.cat(all_z)
cr_train = torch.cat(all_cr)
patch_train = torch.cat(all_patches)
print("  %d training patches" % len(z_train), flush=True)

z_flat = z_train.reshape(len(z_train), -1)
mins_g = z_flat.min(dim=0)[0]
maxs_g = z_flat.max(dim=0)[0]


# ================================================================
# PD quantization + predictor training
# ================================================================
T = 4
BIT_SCHEDULE = [10, 8, 6, 4, 2]
LEVELS = [2**b for b in BIT_SCHEDULE]

def pd_quantize(z, t):
    z_flat = z.reshape(z.shape[0], -1)
    nl = LEVELS[t]
    step = (maxs_g - mins_g) / max(nl - 1, 1)
    q = torch.round((z_flat - mins_g) / step).clamp(0, nl - 1)
    return (mins_g + q * step).reshape_as(z)

print("\n[3] Training PD predictor...", flush=True)

# Precompute quantized levels
z_q_levels = {}
for t in range(T + 1):
    z_q_levels[t] = pd_quantize(z_train, t).detach()

predictor = PDPredictor(D_LAT, 1024, 4).to(device)
opt_p = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-5)
sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=200, eta_min=1e-6)
N = len(z_train); bs = 512

for ep in range(200):
    perm = torch.randperm(N)
    for i in range(0, N, bs):
        idx = perm[i:i+bs]
        z0 = z_train[idx]
        t_val = np.random.randint(1, T + 1)
        z_t = z_q_levels[t_val][idx]
        tn = torch.full((len(idx), 1), t_val / T, device=device)
        pred = predictor(z_t.reshape(len(idx), -1), tn)
        loss = F.mse_loss(pred, (z0 - z_t).reshape(len(idx), -1))
        opt_p.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        opt_p.step()
    sched_p.step()
predictor.eval()
print("  Done.", flush=True)


# ================================================================
# PD recovery function
# ================================================================
def pd_recover(z_coarse, t_start, predictor, schedule):
    z = z_coarse.clone()
    B = z.shape[0]
    for t_target, alpha, n_iters in schedule:
        tn = torch.full((B, 1), t_target / T, device=device)
        for _ in range(n_iters):
            z_flat = z.reshape(B, -1)
            pred = predictor(z_flat, tn).reshape_as(z)
            z = z + alpha * pred
            z = z.clamp(mins_g.min(), maxs_g.max())
    return z

# Schedules (from v1, Assertion-1-guided)
SCHEDULE_T4 = [(4, 0.3, 5), (3, 0.2, 5), (2, 0.1, 3), (1, 0.05, 2)]
SCHEDULE_T2 = [(2, 0.2, 5), (1, 0.1, 3)]
SCHEDULE_T3 = [(3, 0.2, 5), (2, 0.1, 3), (1, 0.05, 2)]


# ================================================================
# Generate PD-recovered training data for decoder fine-tuning
# ================================================================
print("\n[4] Generating PD-recovered latents for decoder training...", flush=True)

# Three recovery configs to fine-tune for
recovery_configs = {
    't4': {'t_start': 4, 'schedule': SCHEDULE_T4, 'bits': BIT_SCHEDULE[4]},
    't3': {'t_start': 3, 'schedule': SCHEDULE_T3, 'bits': BIT_SCHEDULE[3]},
    't2': {'t_start': 2, 'schedule': SCHEDULE_T2, 'bits': BIT_SCHEDULE[2]},
}

# Precompute PD-recovered latents for training (in batches)
z_recovered_train = {}
with torch.no_grad():
    for name, cfg in recovery_configs.items():
        print("  Computing PD recovery: %s (t_start=%d)..." % (name, cfg['t_start']), flush=True)
        recovered = []
        for i in range(0, N, 256):
            z_batch = z_train[i:i+256]
            z_coarse = pd_quantize(z_batch, cfg['t_start'])
            z_rec = pd_recover(z_coarse, cfg['t_start'], predictor, cfg['schedule'])
            recovered.append(z_rec)
        z_recovered_train[name] = torch.cat(recovered)

        # Latent MSE
        mse_before = F.mse_loss(pd_quantize(z_train, cfg['t_start']), z_train).item()
        mse_after = F.mse_loss(z_recovered_train[name], z_train).item()
        print("    Latent MSE: before=%.6f → after=%.6f (%.1f%% recovery)" % (
            mse_before, mse_after, (1 - mse_after/mse_before)*100 if mse_before > 0 else 0), flush=True)


# ================================================================
# Fine-tune decoder for each PD recovery config
# ================================================================
print("\n[5] Fine-tuning decoders...", flush=True)

def make_decoder():
    dec = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
    dec.load_state_dict(vd['dec'])
    return dec

finetuned_decoders = {}

for name, cfg in recovery_configs.items():
    print("\n  Fine-tuning decoder for PD_%s..." % name, flush=True)
    dec = make_decoder()
    opt = torch.optim.AdamW(dec.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-6)

    z_rec = z_recovered_train[name]
    for ep in range(30):
        dec.train()
        total = 0; nb = 0
        perm = torch.randperm(N)
        for i in range(0, N, 256):
            idx = perm[i:i+256]
            batch_patches = patch_train[idx].to(device)
            batch_cr = cr_train[idx].to(device)
            batch_z_rec = z_rec[idx]

            with torch.no_grad():
                target = batch_patches - batch_cr
            pred = dec(batch_z_rec)
            loss = F.mse_loss(pred, target) + 0.2 * F.l1_loss(pred, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if (ep+1) % 10 == 0:
            print("    Epoch %2d: loss=%.6f" % (ep+1, total/nb), flush=True)
    dec.eval()
    finetuned_decoders[name] = dec


# ================================================================
# Evaluate
# ================================================================
print("\n[6] Evaluating on %d test images..." % len(eval_paths), flush=True)

def reconstruct_image_eval(t_img, mode, decoder=None, t_start=None, schedule=None):
    _, H, W = t_img.shape
    ph, pw = (P-H%P)%P, (P-W%P)%P
    t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
    nH, nW = t_pad.shape[1]//P, t_pad.shape[2]//P
    N_p = nH * nW
    patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)
    dec_grid = torch.zeros(nH, nW, 3, P, P, device=device)

    with torch.no_grad():
        patches_gpu = patches.to(device)
        ct = codec.coarse_encoder(patches_gpu)
        cr = codec.coarse_decoder(ct, coords)
        z0 = sp_enc(torch.cat([patches_gpu, cr], dim=1))

        if mode == 'uniform8b':
            z_dq = pd_quantize(z0, 1)
            res = sp_dec_orig(z_dq)
            bits_per_val = 8
        elif mode == 'uniform6b':
            z_dq = pd_quantize(z0, 2)
            res = sp_dec_orig(z_dq)
            bits_per_val = 6
        elif mode == 'uniform4b':
            z_dq = pd_quantize(z0, 3)
            res = sp_dec_orig(z_dq)
            bits_per_val = 4
        elif mode == 'uniform2b':
            z_dq = pd_quantize(z0, 4)
            res = sp_dec_orig(z_dq)
            bits_per_val = 2
        elif mode == 'pd_t4_norecovery':
            z_dq = pd_quantize(z0, 4)
            res = sp_dec_orig(z_dq)
            bits_per_val = 2
        elif mode == 'pd_t4_recovery_orig':
            z_coarse = pd_quantize(z0, 4)
            z_rec = pd_recover(z_coarse, 4, predictor, SCHEDULE_T4)
            res = sp_dec_orig(z_rec)
            bits_per_val = 2
        elif mode == 'pd_t4_recovery_ft':
            z_coarse = pd_quantize(z0, 4)
            z_rec = pd_recover(z_coarse, 4, predictor, SCHEDULE_T4)
            res = finetuned_decoders['t4'](z_rec)
            bits_per_val = 2
        elif mode == 'pd_t3_recovery_ft':
            z_coarse = pd_quantize(z0, 3)
            z_rec = pd_recover(z_coarse, 3, predictor, SCHEDULE_T3)
            res = finetuned_decoders['t3'](z_rec)
            bits_per_val = 4
        elif mode == 'pd_t2_recovery_ft':
            z_coarse = pd_quantize(z0, 2)
            z_rec = pd_recover(z_coarse, 2, predictor, SCHEDULE_T2)
            res = finetuned_decoders['t2'](z_rec)
            bits_per_val = 6
        else:
            raise ValueError(mode)

        finals = (cr + res).clamp(0, 1)
        for idx in range(N_p):
            i2, j2 = idx // nW, idx % nW
            dec_grid[i2, j2] = finals[idx]

    result = dec_grid.permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]
    total_bits = N_p * codec.D1 * 32 + N_p * LAT_SP**2 * bits_per_val * LAT_CH
    bpp = total_bits / (H * W)
    return result.clamp(0, 1), bpp


test_modes = [
    ('uniform8b',              'Uniform 8b (baseline)'),
    ('uniform6b',              'Uniform 6b (one-shot)'),
    ('uniform4b',              'Uniform 4b (one-shot)'),
    ('uniform2b',              'Uniform 2b (one-shot)'),
    ('pd_t4_norecovery',       '2b no recovery (one-shot)'),
    ('pd_t4_recovery_orig',    '2b + PD recovery (orig decoder)'),
    ('pd_t4_recovery_ft',      '2b + PD recovery + decoder FT'),
    ('pd_t3_recovery_ft',      '4b + PD recovery + decoder FT'),
    ('pd_t2_recovery_ft',      '6b + PD recovery + decoder FT'),
]

results = {}
for img_idx, path in enumerate(eval_paths):
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)

    for mode, label in test_modes:
        rec, bpp = reconstruct_image_eval(t_img, mode)
        mse = np.mean((inp - rec.cpu().permute(1,2,0).numpy())**2)
        psnr = 20 * math.log10(1.0 / max(math.sqrt(mse), 1e-10))
        if mode not in results:
            results[mode] = {'label': label, 'psnrs': [], 'bpps': []}
        results[mode]['psnrs'].append(psnr)
        results[mode]['bpps'].append(bpp)

    # Save images for first 3
    if img_idx < 3:
        Image.fromarray((inp*255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'v2_%d_00original.png' % img_idx))
        for si, (mode, label) in enumerate(test_modes):
            rec, _ = reconstruct_image_eval(t_img, mode)
            tag = mode.replace('_','')
            p = results[mode]['psnrs'][img_idx]
            Image.fromarray((rec.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)).save(
                os.path.join(OUTPUT_DIR, 'v2_%d_%02d%s_%.1fdB.png' % (img_idx, si+1, tag, p)))

    print("  %d/%d" % (img_idx+1, len(eval_paths)), flush=True)


# ================================================================
# Results
# ================================================================
print("\n" + "=" * 76)
print("  PD RECONSTRUCTION v2 RESULTS (%d images)" % len(eval_paths))
print("=" * 76)

print("\n  %-45s %7s %7s %7s" % ("Configuration", "PSNR", "BPP", "dB vs U8"))
print("  " + "-" * 68)

u8_psnr = np.mean(results['uniform8b']['psnrs'])

summary = {}
for mode, label in test_modes:
    r = results[mode]
    psnr = np.mean(r['psnrs'])
    bpp = np.mean(r['bpps'])
    diff = psnr - u8_psnr
    print("  %-45s %6.2f %7.2f %+7.2f" % (label, psnr, bpp, diff))
    summary[mode] = {'label': label, 'psnr': float(psnr), 'bpp': float(bpp)}

# Improvement summary
print("\n  PD Recovery + Decoder FT Improvement Chain:")
v_no = np.mean(results['pd_t4_norecovery']['psnrs'])
v_rec = np.mean(results['pd_t4_recovery_orig']['psnrs'])
v_ft = np.mean(results['pd_t4_recovery_ft']['psnrs'])
print("    2b no recovery:          %.2f dB" % v_no)
print("    2b + PD recovery:        %.2f dB  (%+.2f)" % (v_rec, v_rec - v_no))
print("    2b + PD + decoder FT:    %.2f dB  (%+.2f)" % (v_ft, v_ft - v_rec))
print("    Total gain from PD:      %.2f dB  (%.2f → %.2f)" % (v_ft - v_no, v_no, v_ft))

print("\n" + "=" * 76)

with open(os.path.join(OUTPUT_DIR, 'pd_reconstruction_v2.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("  Results: %s" % OUTPUT_DIR)
print("=" * 76)
