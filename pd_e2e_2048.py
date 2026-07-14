#!/usr/bin/env python3
"""
End-to-End PD vs STE at d=2048: Where PD Pulls Ahead
=====================================================
d=8 was too easy — STE's fake gradient was "good enough".
Now: full 2048-dim latent, aggressive K sweep {256, 64, 16}.

Hypothesis: as K decreases (more aggressive quantization),
STE's identity-gradient approximation degrades,
while PD's real gradient maintains quality.

Three configs × three K values = 9 experiments:
  A: Frozen encoder (baseline)
  B: STE end-to-end
  C: PD end-to-end
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_e2e_2048')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')

K_VALUES = [256, 64, 16]

print("=" * 76)
print("  End-to-End PD vs STE at d=2048: Aggressive Quantization Sweep")
print("  K = %s | d = 2048" % K_VALUES)
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
    def __init__(self, d, hidden=512, n_res=2):
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

print("\n[1] Loading v10 codec...", flush=True)
ckpt = torch.load(os.path.join(CKPT_DIR, 'codec_residual.pt'), map_location=device, weights_only=False)
codec = ResidualCodec(K1=ckpt['K1'], K2=ckpt['K2'], P=32).to(device)
codec.load_state_dict(ckpt['model']); codec.eval()
for p in codec.parameters(): p.requires_grad = False
coords = codec._c(device)
P = codec.P

v10c = torch.load(os.path.join(V10_DIR, 'spatial_residual_models_v10.pt'), map_location=device, weights_only=False)
vd = v10c['r4_clean']
LAT_CH, LAT_DS = vd['ch'], vd['ds']
LAT_SP = P // LAT_DS
D_LAT = LAT_CH * LAT_SP ** 2  # 2048

print("\n[2] Preparing data...", flush=True)
paths = find_mini_imagenet(max_count=5000)
np.random.seed(42); np.random.shuffle(paths)
train_paths = paths[:120]
eval_paths = paths[120:132]

def precompute(path_list):
    all_p = []; all_cr = []
    for path in path_list:
        try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
        except: continue
        inp = np.array(img, dtype=np.float32) / 255.0
        t_img = torch.from_numpy(inp).permute(2,0,1)
        patches = t_img.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)
        with torch.no_grad():
            ct = codec.coarse_encoder(patches.to(device))
            cr = codec.coarse_decoder(ct, coords)
        all_p.append(patches); all_cr.append(cr.cpu())
    return torch.cat(all_p), torch.cat(all_cr)

train_patches, train_cr = precompute(train_paths)
eval_patches, eval_cr = precompute(eval_paths)
print("  Train: %d | Eval: %d" % (len(train_patches), len(eval_patches)), flush=True)


# ================================================================
# VQ layer (full 2048-dim)
# ================================================================
class VQCodebook(nn.Module):
    def __init__(self, K, d):
        super().__init__()
        self.K = K
        self.d = d
        self.codebook = nn.Parameter(torch.randn(K, d) * 0.05)

    def forward_ste(self, z_flat):
        """STE: z_flat (B, 2048) → z_q with straight-through gradient."""
        with torch.no_grad():
            d = torch.cdist(z_flat, self.codebook)
            indices = d.argmin(dim=1)
            z_q_hard = self.codebook[indices]
        # STE: gradient = identity
        z_q = z_flat + (z_q_hard - z_flat).detach()
        commit = F.mse_loss(z_flat, z_q_hard.detach())
        return z_q, z_q_hard, commit

    def forward_pd(self, z_flat, pd_pred, t_val=0.5):
        """PD: z_flat (B, 2048) → z_q with predictor-corrected gradient."""
        with torch.no_grad():
            d = torch.cdist(z_flat, self.codebook)
            indices = d.argmin(dim=1)
            z_q_hard = self.codebook[indices]

        # PD predictor: estimates z_0 - z_q (the quantization residual)
        B = z_flat.shape[0]
        t_norm = torch.full((B, 1), t_val, device=z_flat.device)
        pd_correction = pd_pred(z_q_hard.detach(), t_norm)

        # Blend: STE path for encoder gradient + PD correction for quality
        z_q_pd = z_q_hard + pd_correction
        # STE wrapper: encoder gets gradient via z_flat, codebook+predictor via pd_correction
        z_q = z_flat + (z_q_pd - z_flat).detach()
        commit = F.mse_loss(z_flat, z_q_hard.detach())
        pd_loss = F.mse_loss(pd_correction, (z_flat - z_q_hard).detach())
        return z_q, z_q_hard, commit, pd_loss


def init_codebook(K, train_data):
    """K-means init codebook."""
    cb = VQCodebook(K, D_LAT).to(device)
    with torch.no_grad():
        # Sample init
        idx = torch.randperm(len(train_data))[:K]
        cb.codebook.data = train_data[idx].clone()
        # Lloyd iterations
        for _ in range(15):
            assign = torch.zeros(len(train_data), dtype=torch.long, device=device)
            for i in range(0, len(train_data), 2048):
                d = torch.cdist(train_data[i:i+2048], cb.codebook)
                assign[i:i+2048] = d.argmin(dim=1)
            for k in range(K):
                mask = assign == k
                if mask.sum() > 0:
                    cb.codebook.data[k] = train_data[mask].mean(dim=0)
    return cb


def make_encoder():
    enc = ResBlockEnc(vd['ch'], vd['ds'], vd.get('n_res',4)).to(device)
    enc.load_state_dict(vd['enc'])
    return enc

def make_decoder():
    dec = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
    dec.load_state_dict(vd['dec'])
    return dec


# ================================================================
# Training function
# ================================================================
def train_model(mode, K, epochs=40):
    """mode: 'frozen', 'ste', 'pd'"""
    label = "%s_K%d" % (mode.upper(), K)
    print("\n  Training %s ..." % label, flush=True)

    enc = make_encoder()
    dec = make_decoder()

    # Get initial latents for codebook init
    enc.eval()
    with torch.no_grad():
        z_list = []
        for i in range(0, min(len(train_patches), 2560), 256):
            z = enc(torch.cat([train_patches[i:i+256].to(device), train_cr[i:i+256].to(device)], dim=1))
            z_list.append(z.reshape(len(z), -1))
        z_init = torch.cat(z_list)

    vq = init_codebook(K, z_init)
    pd_pred = PDPredictor(D_LAT, hidden=512, n_res=2).to(device)

    # Freeze/unfreeze encoder
    if mode == 'frozen':
        for p in enc.parameters(): p.requires_grad = False
        enc.eval()
        params = list(dec.parameters()) + list(vq.parameters())
    elif mode == 'ste':
        params = list(enc.parameters()) + list(dec.parameters()) + list(vq.parameters())
    elif mode == 'pd':
        params = list(enc.parameters()) + list(dec.parameters()) + list(vq.parameters()) + list(pd_pred.parameters())

    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    N = len(train_patches)
    best_psnr = 0

    for ep in range(epochs):
        if mode != 'frozen': enc.train()
        dec.train()
        if mode == 'pd': pd_pred.train()

        perm = torch.randperm(N)
        total = 0; nb = 0
        for i in range(0, N, 128):
            idx = perm[i:i+128]
            p = train_patches[idx].to(device)
            c = train_cr[idx].to(device)

            z = enc(torch.cat([p, c], dim=1))  # (B, 8, 16, 16)
            z_flat = z.reshape(len(p), -1)  # (B, 2048)

            if mode == 'pd':
                z_q_flat, z_q_hard, commit, pd_loss = vq.forward_pd(z_flat, pd_pred)
            else:
                z_q_flat, z_q_hard, commit = vq.forward_ste(z_flat)
                pd_loss = 0

            z_q = z_q_flat.reshape(len(p), LAT_CH, LAT_SP, LAT_SP)
            res = dec(z_q)
            target = p - c
            recon = F.mse_loss(res, target) + 0.2 * F.l1_loss(res, target)
            loss = recon + 0.25 * commit + (0.1 * pd_loss if mode == 'pd' else 0)

            opt.zero_grad(); loss.backward()
            if mode != 'frozen':
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1

        sched.step()

        if (ep+1) % 10 == 0 or ep == 0:
            # Eval
            enc.eval(); dec.eval()
            psnrs = []
            with torch.no_grad():
                for ei in range(0, min(len(eval_patches), 256), 128):
                    ep_ = eval_patches[ei:ei+128].to(device)
                    ec_ = eval_cr[ei:ei+128].to(device)
                    z = enc(torch.cat([ep_, ec_], dim=1)).reshape(len(ep_), -1)
                    z_q, _, _ = vq.forward_ste(z)
                    z_q4d = z_q.reshape(len(ep_), LAT_CH, LAT_SP, LAT_SP)
                    res = dec(z_q4d)
                    final = (ec_ + res).clamp(0, 1)
                    for j in range(len(ep_)):
                        mse = F.mse_loss(final[j], ep_[j])
                        psnrs.append(20*math.log10(1/max(math.sqrt(mse.item()),1e-10)))
            avg_psnr = np.mean(psnrs)
            best_psnr = max(best_psnr, avg_psnr)
            print("    [%s] Epoch %2d: loss=%.4f  PSNR=%.2f dB" % (label, ep+1, total/nb, avg_psnr), flush=True)

    # Final eval on full set
    enc.eval(); dec.eval()
    psnrs = []
    with torch.no_grad():
        for ei in range(0, len(eval_patches), 128):
            ep_ = eval_patches[ei:ei+128].to(device)
            ec_ = eval_cr[ei:ei+128].to(device)
            z = enc(torch.cat([ep_, ec_], dim=1)).reshape(len(ep_), -1)
            z_q, _, _ = vq.forward_ste(z)
            z_q4d = z_q.reshape(len(ep_), LAT_CH, LAT_SP, LAT_SP)
            res = dec(z_q4d)
            final = (ec_ + res).clamp(0, 1)
            for j in range(len(ep_)):
                mse = F.mse_loss(final[j], ep_[j])
                psnrs.append(20*math.log10(1/max(math.sqrt(mse.item()),1e-10)))

    final_psnr = np.mean(psnrs)

    # Measure latent adaptation
    with torch.no_grad():
        z_sample = enc(torch.cat([eval_patches[:64].to(device), eval_cr[:64].to(device)], dim=1))
        z_flat = z_sample.reshape(-1)
        z_range = (z_sample.max() - z_sample.min()).item()
        z_std = z_flat.std().item()
        # VQ MSE
        z_flat_2 = z_sample.reshape(64, -1)
        d_vq = torch.cdist(z_flat_2, vq.codebook)
        z_q = vq.codebook[d_vq.argmin(dim=1)]
        vq_mse = F.mse_loss(z_q, z_flat_2).item()

    print("    [%s] FINAL: PSNR=%.2f dB  z_range=%.3f  vq_mse=%.6f" % (
        label, final_psnr, z_range, vq_mse), flush=True)

    return {
        'psnr': final_psnr,
        'z_range': z_range,
        'z_std': z_std,
        'vq_mse': vq_mse,
    }


# ================================================================
# Run all experiments
# ================================================================
print("\n[3] Running experiments...", flush=True)

all_results = {}
for K in K_VALUES:
    print("\n" + "=" * 60, flush=True)
    print("  K = %d" % K, flush=True)
    print("=" * 60, flush=True)

    all_results[K] = {}
    for mode in ['frozen', 'ste', 'pd']:
        t0 = time.time()
        result = train_model(mode, K, epochs=40)
        result['time'] = time.time() - t0
        all_results[K][mode] = result
        print("  Time: %.0fs\n" % result['time'], flush=True)


# ================================================================
# Results
# ================================================================
print("\n" + "=" * 76)
print("  END-TO-END RESULTS: PD vs STE vs Frozen at d=2048")
print("=" * 76)

print("\n  PSNR (dB) by K value:")
print("  %-12s %8s %8s %8s" % ("Method", "K=256", "K=64", "K=16"))
print("  " + "-" * 40)
for mode in ['frozen', 'ste', 'pd']:
    vals = [all_results[K][mode]['psnr'] for K in K_VALUES]
    print("  %-12s %7.2f %7.2f %7.2f" % (mode.upper(), *vals))

print("\n  PD vs STE (dB):")
print("  %-12s %8s %8s %8s" % ("", "K=256", "K=64", "K=16"))
print("  " + "-" * 40)
for K in K_VALUES:
    diff = all_results[K]['pd']['psnr'] - all_results[K]['ste']['psnr']
    print("  K=%-9d %+7.2f" % (K, diff) if K == K_VALUES[0] else "", end="")
print()
for K in K_VALUES:
    diff = all_results[K]['pd']['psnr'] - all_results[K]['ste']['psnr']
    print("  PD - STE @ K=%-4d: %+.2f dB %s" % (
        K, diff, "← PD WINS" if diff > 0 else "← STE wins"))

print("\n  Encoder Adaptation (z_range, lower=more compact):")
print("  %-12s %8s %8s %8s" % ("Method", "K=256", "K=64", "K=16"))
print("  " + "-" * 40)
for mode in ['frozen', 'ste', 'pd']:
    vals = [all_results[K][mode]['z_range'] for K in K_VALUES]
    print("  %-12s %7.3f %7.3f %7.3f" % (mode.upper(), *vals))

print("\n  VQ MSE (latent quantization error):")
print("  %-12s %8s %8s %8s" % ("Method", "K=256", "K=64", "K=16"))
print("  " + "-" * 40)
for mode in ['frozen', 'ste', 'pd']:
    vals = [all_results[K][mode]['vq_mse'] for K in K_VALUES]
    print("  %-12s %7.6f %7.6f %7.6f" % (mode.upper(), *vals))

print("\n  Key finding:")
k256_diff = all_results[256]['pd']['psnr'] - all_results[256]['ste']['psnr']
k64_diff = all_results[64]['pd']['psnr'] - all_results[64]['ste']['psnr']
k16_diff = all_results[16]['pd']['psnr'] - all_results[16]['ste']['psnr']
print("    K=256: PD %+.2f dB vs STE" % k256_diff)
print("    K=64:  PD %+.2f dB vs STE" % k64_diff)
print("    K=16:  PD %+.2f dB vs STE" % k16_diff)
if k16_diff > k64_diff > k256_diff:
    print("    → PD advantage GROWS as K decreases (hypothesis confirmed!)")
elif k16_diff > k256_diff:
    print("    → PD advantage is larger at smaller K")
else:
    print("    → PD ≈ STE even at small K (d=2048 is still tractable for STE)")

print("\n" + "=" * 76)

# Save
save_data = {}
for K in K_VALUES:
    save_data[str(K)] = {}
    for mode in ['frozen', 'ste', 'pd']:
        save_data[str(K)][mode] = {k: float(v) for k, v in all_results[K][mode].items()}
with open(os.path.join(OUTPUT_DIR, 'e2e_2048_results.json'), 'w') as f:
    json.dump(save_data, f, indent=2)
print("  Saved: %s" % OUTPUT_DIR)
print("=" * 76)
