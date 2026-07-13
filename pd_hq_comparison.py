#!/usr/bin/env python3
"""
High-Quality RVQ vs Uniform Comparison (0% prediction, decoder fine-tuning)
===========================================================================
Previous issue: simplified prediction gave only ~15 dB.
Fix: use 0% prediction (all anchors) to isolate quantization quality.

Pipeline:
  1. Reproduce v10 baseline: uniform 8b, 0% pred → should get ~40 dB
  2. Fine-tune ResBlockDec for each RVQ configuration
  3. Compare rate-quality: uniform {4,6,8}b vs RVQ {2,3,4} stages

With 0% prediction:
  BPP = D1*32/(32*32) + bits_residual/(32*32)
  Uniform 8b:  24.09 + 16.00 = 40.09 BPP  (v10 baseline ~40 dB)
  RVQ 4-stage: 24.09 +  8.00 = 32.09 BPP  (20% less BPP if quality holds)
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_hq_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')

print("=" * 76)
print("  High-Quality RVQ vs Uniform (0% pred + decoder fine-tuning)")
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

# Load
print("\n[1] Loading models...", flush=True)
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
sp_dec_orig = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
sp_dec_orig.load_state_dict(vd['dec']); sp_dec_orig.eval()
ch_mins = vd['ch_mins'].to(device)
ch_maxs = vd['ch_maxs'].to(device)
LAT_CH, LAT_DS = vd['ch'], vd['ds']
LAT_SP = P // LAT_DS
N_POS = LAT_SP ** 2
print("  Loaded. ch=%d ds=%d" % (LAT_CH, LAT_DS), flush=True)


# ================================================================
# Quantization functions
# ================================================================
def quant_uniform(z, mins, maxs, bits):
    levels = 2 ** bits
    step = (maxs - mins) / max(levels - 1, 1)
    mv = mins.view(1, -1, 1, 1); sv = step.view(1, -1, 1, 1)
    q = torch.round((z - mv) / sv).clamp(0, levels - 1)
    return mv + q * sv

def rvq_encode(z, codebooks):
    """z: (B, 8, 16, 16) → list of (B, 256) index tensors."""
    B = z.shape[0]
    z_pos = z.permute(0, 2, 3, 1).reshape(B, -1, LAT_CH)  # (B, 256, 8)
    residual = z_pos
    indices = []
    for cb in codebooks:
        dists = torch.cdist(residual, cb)  # (B, 256, K)
        idx = dists.argmin(dim=2)  # (B, 256)
        indices.append(idx)
        residual = residual - cb[idx]
    return indices

def rvq_decode(indices, codebooks):
    """List of (B, 256) → (B, 8, 16, 16)"""
    B = indices[0].shape[0]
    z_pos = torch.zeros(B, N_POS, LAT_CH, device=indices[0].device)
    for s, idx in enumerate(indices):
        z_pos += codebooks[s][idx]
    return z_pos.permute(0, 2, 1).reshape(B, LAT_CH, LAT_SP, LAT_SP)


# ================================================================
# Extract latents + patches for training/eval
# ================================================================
print("\n[2] Extracting latents...", flush=True)
paths = find_mini_imagenet(max_count=5000)
np.random.seed(42); np.random.shuffle(paths)
train_paths = paths[:60]
eval_paths = paths[60:75]

def process_images(path_list):
    """Returns patches (N,3,32,32), latents (N,8,16,16), coarse_recon (N,3,32,32)."""
    all_patches = []; all_z = []; all_cr = []
    for path in path_list:
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
        all_patches.append(patches)
        all_z.append(z.cpu())
        all_cr.append(cr.cpu())
    return torch.cat(all_patches), torch.cat(all_z), torch.cat(all_cr)

tr_patches, tr_z, tr_cr = process_images(train_paths)
print("  Train: %d patches" % len(tr_z), flush=True)


# ================================================================
# Train RVQ codebooks
# ================================================================
print("\n[3] Training RVQ codebooks...", flush=True)
pos_data = tr_z.permute(0, 2, 3, 1).reshape(-1, LAT_CH).numpy()
# Subsample
rng_s = np.random.RandomState(0)
pos_sub = pos_data[rng_s.choice(len(pos_data), min(50000, len(pos_data)), replace=False)]

def kmeans_gpu(data_np, K, iters=40, seed=42):
    dt = torch.from_numpy(data_np).float().to(device)
    rng = np.random.RandomState(seed)
    cb = dt[rng.choice(len(dt), K, replace=False)].clone()
    for it in range(iters):
        assign = torch.zeros(len(dt), dtype=torch.long, device=device)
        for i in range(0, len(dt), 65536):
            dists = torch.cdist(dt[i:i+65536], cb)
            assign[i:i+65536] = dists.argmin(dim=1)
        lr = 1.0 / (1.0 + it * 0.05)
        for k in range(K):
            mask = assign == k
            if mask.sum() > 0:
                cb[k] += lr * (dt[mask].mean(dim=0) - cb[k])
    return cb

K_STAGE = 256
BITS_STAGE = 8
N_STAGES_MAX = 4

rvq_cbs = []
residual_np = pos_sub.copy()
for s in range(N_STAGES_MAX):
    cb = kmeans_gpu(residual_np, K_STAGE, seed=42+s)
    cb_np = cb.cpu().numpy()
    dt = torch.from_numpy(residual_np).float().to(device)
    with torch.no_grad():
        dists = torch.cdist(dt, cb)
        assign = dists.argmin(dim=1)
        q = cb[assign].cpu().numpy()
    residual_np = residual_np - q
    rvq_cbs.append(cb)
    mse = np.mean(residual_np**2)
    print("  Stage %d done: residual MSE=%.6f (%d bits/pos)" % (s, mse, (s+1)*BITS_STAGE), flush=True)

# Convert codebooks to GPU tensors
rvq_cbs_gpu = [cb.clone().to(device) for cb in rvq_cbs]
# Also store as numpy for saving
rvq_cbs_np = [cb.cpu().numpy() for cb in rvq_cbs]


# ================================================================
# Fine-tune decoder for each quantization scheme
# ================================================================
print("\n[4] Fine-tuning decoders for each quantization scheme...", flush=True)

def make_decoder():
    """Create a fresh decoder copy from original weights."""
    dec = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
    dec.load_state_dict(vd['dec'])
    return dec

def finetune_decoder(tr_patches, tr_z, tr_cr, quant_fn, n_epochs=30, lr=5e-4, batch_size=256):
    """Fine-tune decoder for a specific quantization scheme.
    quant_fn: (z_latent) -> dequantized_latent
    """
    dec = make_decoder()
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    N = len(tr_z)
    for ep in range(n_epochs):
        dec.train()
        total_loss = 0; nb = 0
        for i in range(0, N, batch_size):
            batch_patches = tr_patches[i:i+batch_size].to(device)
            batch_z = tr_z[i:i+batch_size].to(device)
            batch_cr = tr_cr[i:i+batch_size].to(device)

            with torch.no_grad():
                z_dq = quant_fn(batch_z)
                target = batch_patches - batch_cr  # residual

            pred = dec(z_dq)
            loss = F.mse_loss(pred, target) + 0.2 * F.l1_loss(pred, target)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); nb += 1

        sched.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print("    Epoch %2d/%d: loss=%.6f" % (ep+1, n_epochs, total_loss/nb), flush=True)

    dec.eval()
    return dec

# Define quantization schemes
schemes = {}

# Uniform schemes (no fine-tuning needed, use original decoder)
for bits in [4, 6, 8]:
    name = "Uniform_%db" % bits
    schemes[name] = {
        'type': 'uniform', 'bits': bits,
        'decoder': sp_dec_orig,  # original decoder, no fine-tuning
        'bits_per_pos': bits * LAT_CH,
    }

# RVQ schemes (fine-tune decoder for each stage count)
for n_stages in [2, 3, 4]:
    name = "RVQ_%dstages" % n_stages
    print("\n  Fine-tuning decoder for %s..." % name, flush=True)

    cbs = rvq_cbs_gpu[:n_stages]

    def make_quant_fn(cbs_list):
        def quant_fn(z):
            indices = rvq_encode(z, cbs_list)
            return rvq_decode(indices, cbs_list)
        return quant_fn

    dec_ft = finetune_decoder(tr_patches, tr_z, tr_cr, make_quant_fn(cbs), n_epochs=25, lr=5e-4)

    schemes[name] = {
        'type': 'rvq', 'n_stages': n_stages,
        'decoder': dec_ft,
        'bits_per_pos': n_stages * BITS_STAGE,
    }


# ================================================================
# Evaluate on test images (0% prediction = all anchors)
# ================================================================
print("\n[5] Evaluating on %d test images (0%% prediction)..." % len(eval_paths), flush=True)

results = {}
base_bpp = (codec.D1 + codec.D2) * 32 / (32*32)  # original codec BPP

for scheme_name, scheme in schemes.items():
    print("  Testing %s..." % scheme_name, flush=True)
    psnrs = []; bpps = []

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

        dec_grid = torch.zeros(nH, nW, 3, P, P, device=device)

        with torch.no_grad():
            # Batch process all patches for speed
            patches_gpu = patches.to(device)
            ct = codec.coarse_encoder(patches_gpu)
            cr = codec.coarse_decoder(ct, coords)
            z = sp_enc(torch.cat([patches_gpu, cr], dim=1))

            if scheme['type'] == 'uniform':
                z_dq = quant_uniform(z, ch_mins, ch_maxs, scheme['bits'])
            else:
                cbs = rvq_cbs_gpu[:scheme['n_stages']]
                indices = rvq_encode(z, cbs)
                z_dq = rvq_decode(indices, cbs)

            res = scheme['decoder'](z_dq)
            finals = (cr + res).clamp(0, 1)

            for idx in range(N):
                i, j = idx // nW, idx % nW
                dec_grid[i, j] = finals[idx]

        result = dec_grid.permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]
        mse = np.mean((inp - result.cpu().permute(1,2,0).numpy())**2)
        psnr = 20 * math.log10(1.0 / max(math.sqrt(mse), 1e-10))
        psnrs.append(psnr)

        bits_per_patch = N_POS * scheme['bits_per_pos']
        total_bits = N * codec.D1 * 32 + N * bits_per_patch
        bpp = total_bits / (H * W)
        bpps.append(bpp)

    avg_psnr = np.mean(psnrs)
    avg_bpp = np.mean(bpps)
    results[scheme_name] = {
        'psnr': avg_psnr, 'bpp': avg_bpp,
        'psnrs': psnrs, 'bpps': bpps,
        'bits_per_pos': scheme['bits_per_pos'],
        'compr': base_bpp / avg_bpp,
    }
    print("    PSNR=%.2f dB, BPP=%.2f (%.1fx)" % (avg_psnr, avg_bpp, base_bpp/avg_bpp), flush=True)


# ================================================================
# Save comparison images
# ================================================================
print("\n[6] Saving comparison images...", flush=True)

for img_idx in range(3):
    path = eval_paths[img_idx]
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)

    # Original
    Image.fromarray((inp*255).astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, 'cmp%d_00original.png' % img_idx))

    for si, (scheme_name, scheme) in enumerate(schemes.items()):
        _, H, W = t_img.shape
        ph, pw = (P-H%P)%P, (P-W%P)%P
        t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
        nH, nW = t_pad.shape[1]//P, t_pad.shape[2]//P
        patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)

        with torch.no_grad():
            patches_gpu = patches.to(device)
            ct = codec.coarse_encoder(patches_gpu)
            cr = codec.coarse_decoder(ct, coords)
            z = sp_enc(torch.cat([patches_gpu, cr], dim=1))
            if scheme['type'] == 'uniform':
                z_dq = quant_uniform(z, ch_mins, ch_maxs, scheme['bits'])
            else:
                cbs = rvq_cbs_gpu[:scheme['n_stages']]
                indices = rvq_encode(z, cbs)
                z_dq = rvq_decode(indices, cbs)
            res = scheme['decoder'](z_dq)
            finals = (cr + res).clamp(0, 1)

        N = nH * nW
        dec_grid = torch.zeros(nH, nW, 3, P, P, device=device)
        for idx in range(N):
            i2, j2 = idx // nW, idx % nW
            dec_grid[i2, j2] = finals[idx]
        result = dec_grid.permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]
        rec_np = result.cpu().permute(1,2,0).numpy()

        tag = scheme_name.replace('_','')
        psnr_val = results[scheme_name]['psnrs'][img_idx]
        Image.fromarray((rec_np*255).clip(0,255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'cmp%d_%02d%s_%.1fdB.png' % (img_idx, si+1, tag, psnr_val)))


# ================================================================
# RESULTS
# ================================================================
print("\n" + "=" * 76)
print("  HIGH-QUALITY RESULTS: RVQ vs Uniform (0%% pred, %d images)" % len(eval_paths))
print("=" * 76)

print("\n  %-25s %7s %8s %7s %7s %8s" % ("Scheme", "PSNR", "bits/pos", "BPP", "Compr", "dB vs U8"))
print("  " + "-" * 68)

u8_psnr = results.get('Uniform_8b', {}).get('psnr', 0)

for name in ['Uniform_4b', 'Uniform_6b', 'Uniform_8b', 'RVQ_2stages', 'RVQ_3stages', 'RVQ_4stages']:
    if name not in results: continue
    r = results[name]
    diff = r['psnr'] - u8_psnr
    print("  %-25s %6.2f %8d %7.2f %6.1fx %+7.2f" % (
        name, r['psnr'], r['bits_per_pos'], r['bpp'], r['compr'], diff))

# Rate-quality analysis
print("\n  Rate-Quality Comparison at similar BPP:")
comparisons = [
    ("~32 BPP", [('Uniform_4b', 32), ('RVQ_4stages', 32)]),
    ("~24 BPP", [('RVQ_3stages', 24)]),
    ("~16 BPP", [('RVQ_2stages', 16)]),
]
for label, items in comparisons:
    print("\n    BPP %s:" % label)
    for name, target_bpp in items:
        if name in results:
            r = results[name]
            print("      %-25s PSNR=%.2f @ %.2f BPP" % (name, r['psnr'], r['bpp']))

print("\n" + "=" * 76)

# Save summary
summary_out = {k: {'psnr': v['psnr'], 'bpp': v['bpp'], 'bits_per_pos': v['bits_per_pos']}
               for k, v in results.items()}
with open(os.path.join(OUTPUT_DIR, 'hq_comparison.json'), 'w') as f:
    json.dump(summary_out, f, indent=2)
print("  Results: %s" % OUTPUT_DIR)
print("  Images: cmp0_00original.png, cmp0_01Uniform8b_*.png, cmp0_04RVQ4stages_*.png, etc.")
print("=" * 76)
