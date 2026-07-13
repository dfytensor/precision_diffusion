#!/usr/bin/env python3
"""
Precision Diffusion v4 — Comprehensive Sweep
=============================================
Full validation: multiple K values, noise schedules, convergence curves.
Addresses: "how close can PD get to K-means at various codebook sizes?"

Sweep:
  K = 16, 64, 256
  Init: K-means (good), Random (scratch)
  Noise: none, fixed-Langevin, adaptive (cosine annealing)
  Iters: up to 200
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_validation_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

T = 4
BIT_SCHEDULE = [10, 8, 6, 4, 2]
LEVELS = [2**b for b in BIT_SCHEDULE]
D_LATENT = 2048
K_VALUES = [16, 64, 256]

beta = np.linspace(0.02, 0.25, T + 1)
alpha = 1.0 - beta
alpha_bar = np.cumprod(alpha)

device = torch.device('cuda')
print("=" * 76)
print("  PD v4 — Comprehensive K-sweep + Noise Schedule Optimization")
print("  d = %d  |  K = %s" % (D_LATENT, K_VALUES))
print("=" * 76)

# ================================================================
# Utils
# ================================================================
def pairwise_sq_dist(A, B):
    aa = np.sum(A**2, axis=1)[:, None]
    bb = np.sum(B**2, axis=1)[None, :]
    return np.maximum(aa - 2*(A @ B.T) + bb, 0)

def assign_centroids(data, centroids, batch=8192):
    n = len(data); result = np.zeros(n, dtype=np.int32)
    for i in range(0, n, batch):
        dists = pairwise_sq_dist(data[i:i+batch], centroids)
        result[i:i+batch] = np.argmin(dists, axis=1)
    return result

def vq_mse(data, centroids):
    n = len(data); total = 0.0
    for i in range(0, n, 8192):
        b = data[i:i+8192]; a = assign_centroids(b, centroids)
        total += np.sum((b - centroids[a])**2)
    return total / n

# ================================================================
# Extract latents (cache to disk)
# ================================================================
cache_path = os.path.join(OUTPUT_DIR, 'latents_cache.npz')
if os.path.exists(cache_path):
    d = np.load(cache_path)
    train_x, test_x = d['train_x'], d['test_x']
    print("  Loaded cached latents: train=%d test=%d" % (len(train_x), len(test_x)))
else:
    print("  Extracting latents from v10 codec...")
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

    from codec_residual import ResidualCodec
    from periodic_token_codec import find_mini_imagenet

    ckpt = torch.load(os.path.join(CKPT_DIR, 'codec_residual.pt'), map_location=device, weights_only=False)
    codec = ResidualCodec(K1=ckpt['K1'], K2=ckpt['K2'], P=32).to(device)
    codec.load_state_dict(ckpt['model']); codec.eval()
    for p in codec.parameters(): p.requires_grad = False
    coords = codec._c(device); P = codec.P

    v10c = torch.load(os.path.join(V10_DIR, 'spatial_residual_models_v10.pt'), map_location=device, weights_only=False)
    mk = 'r4_clean' if 'r4_clean' in v10c else list(v10c.keys())[0]
    vd = v10c[mk]
    enc = ResBlockEnc(vd['ch'], vd['ds'], vd.get('n_res',4)).to(device)
    enc.load_state_dict(vd['enc']); enc.eval()

    paths = find_mini_imagenet(max_count=5000)
    np.random.seed(42); np.random.shuffle(paths)
    all_lat = []
    ct = 0
    for path in paths:
        if ct >= 200: break
        try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
        except: continue
        inp = np.array(img, dtype=np.float32) / 255.0
        t_img = torch.from_numpy(inp).permute(2,0,1)
        H, W = t_img.shape[1], t_img.shape[2]
        ph, pw = (P-H%P)%P, (P-W%P)%P
        t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
        patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)
        with torch.no_grad():
            ct_p = codec.coarse_encoder(patches.to(device))
            cr = codec.coarse_decoder(ct_p, coords)
            z = enc(torch.cat([patches.to(device), cr], dim=1))
        all_lat.append(z.float().cpu().numpy().reshape(len(z), -1))
        ct += 1
    latents = np.concatenate(all_lat, axis=0)
    np.random.seed(0)
    idx = np.random.permutation(len(latents))
    n_tr = min(8000, len(latents)*3//4)
    n_te = min(2000, len(latents) - n_tr)
    train_x, test_x = latents[idx[:n_tr]], latents[idx[n_tr:n_tr+n_te]]
    np.savez(cache_path, train_x=train_x, test_x=test_x)
    print("  Extracted: train=%d test=%d (cached)" % (len(train_x), len(test_x)))
    torch.cuda.empty_cache()

# ================================================================
# Quantizers
# ================================================================
mins_g = train_x.min(axis=0)
maxs_g = train_x.max(axis=0)

def build_q(nl):
    step = (maxs_g - mins_g) / max(nl-1, 1)
    return mins_g, step, nl

def quantize_at(x, qt):
    m, s, nl = qt
    return m + np.round((x-m)/s).clip(0,nl-1).astype(np.float32) * s

quantizers = [build_q(nl) for nl in LEVELS]

# ================================================================
# PD Predictor (shared across all K)
# ================================================================
print("\n  Training PD predictor (12.6M params)...")
class PDPredictor(nn.Module):
    def __init__(self, d, hidden=1024, n_res=4):
        super().__init__()
        self.proj_in = nn.Linear(d+1, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,hidden), nn.GELU(), nn.Linear(hidden,hidden))
            for _ in range(n_res)
        ])
        self.proj_out = nn.Linear(hidden, d)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    def forward(self, x_t, t_norm):
        h = self.proj_in(torch.cat([x_t, t_norm], dim=-1))
        for b in self.blocks: h = h + b(h)
        return self.proj_out(h)

predictor = PDPredictor(D_LATENT, 1024, 4).to(device)
x_torch = torch.from_numpy(train_x).float().to(device)
qt_tensors = [torch.from_numpy(quantize_at(train_x, quantizers[t])).float().to(device) for t in range(1, T+1)]

opt_p = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-5)
sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=150, eta_min=1e-6)
for ep in range(150):
    for i in range(0, len(train_x), 512):
        batch = x_torch[i:i+512]
        t_val = np.random.randint(1, T+1)
        xq = qt_tensors[t_val-1][i:i+512]
        tn = torch.full((len(xq),1), t_val/T, device=device)
        loss = F.mse_loss(predictor(xq, tn), batch - xq)
        opt_p.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        opt_p.step()
    sched_p.step()
predictor.eval()
print("  Done.\n")


# ================================================================
# Training functions
# ================================================================
def kmeans_train(data, K, init=None, iters=100, seed=42, lr_max=0.2):
    rng = np.random.RandomState(seed)
    cb = init.copy() if init is not None else data[rng.choice(len(data), K, replace=False)].copy()
    history = []
    for it in range(iters):
        assign = assign_centroids(data, cb)
        lr = lr_max / (1.0 + it * 0.05)
        for k in range(K):
            mask = assign == k
            if mask.sum() > 0:
                cb[k] += lr * (data[mask].mean(axis=0) - cb[k])
        if it % 20 == 0 or it == iters-1:
            history.append((it, vq_mse(test_x[:1000], cb)))
    return cb, history


def pd_train(data, init_cb, predictor, K, iters, lr, t_match,
             noise_mode='none', temp_start=0.0, schedule='cosine', pred_steps=3):
    """schedule: 'linear', 'cosine', 'step'"""
    cb = torch.from_numpy(init_cb.copy()).float().to(device)
    data_t = torch.from_numpy(data).float().to(device)
    n = len(data)
    opt_pred = torch.optim.AdamW(predictor.parameters(), lr=5e-5, weight_decay=1e-5)
    history = []

    for it in range(iters):
        # Inner: predictor fine-tune
        if pred_steps > 0:
            predictor.train()
            for _ in range(pred_steps):
                idx = torch.randint(0, n, (min(512, n),), device=device)
                batch = data_t[idx]
                with torch.no_grad():
                    dists = torch.cdist(batch, cb)
                    assign = dists.argmin(dim=1)
                    x_q = cb[assign]
                tn = torch.full((len(x_q),1), t_match/T, device=device)
                loss = F.mse_loss(predictor(x_q, tn), batch - x_q)
                opt_pred.zero_grad(); loss.backward(); opt_pred.step()

        # Temperature schedule
        frac = it / max(iters - 1, 1)
        if schedule == 'linear':
            temp = temp_start * max(0, 1.0 - frac)
        elif schedule == 'cosine':
            temp = temp_start * 0.5 * (1 + math.cos(math.pi * frac))
        elif schedule == 'step':
            temp = temp_start if frac < 0.5 else 0
        else:
            temp = 0

        # Outer: codebook update
        predictor.eval()
        with torch.no_grad():
            dists_all = torch.cdist(data_t, cb)
            assign_all = dists_all.argmin(dim=1)
            x_q_all = cb[assign_all]
            tn_all = torch.full((n,1), t_match/T, device=device)
            eps_all = predictor(x_q_all, tn_all)

            for k in range(K):
                mask = assign_all == k
                if mask.sum() > 0:
                    grad = eps_all[mask].mean(dim=0)
                    if noise_mode == 'fixed' and temp > 0:
                        noise = torch.randn_like(cb[k]) * math.sqrt(2*lr*temp/D_LATENT)
                    elif noise_mode == 'adaptive' and temp > 0:
                        gnorm = grad.norm().item() + 1e-8
                        z = torch.randn_like(cb[k]); z = z / (z.norm() + 1e-8)
                        noise = z * temp * gnorm
                    else:
                        noise = 0
                    cb[k] += lr * grad + noise

        if it % 20 == 0 or it == iters-1:
            history.append((it, vq_mse(test_x[:1000], cb.cpu().numpy())))

    return cb.cpu().numpy(), history


# ================================================================
# MAIN SWEEP
# ================================================================
all_results = {}

for K in K_VALUES:
    print("\n" + "=" * 76)
    print("  K = %d" % K)
    print("=" * 76)

    results_K = {}

    # --- K-means ---
    print("  K-means...", end=" ", flush=True)
    t0 = time.time()
    cb_km, hist_km = kmeans_train(train_x, K, iters=100, seed=42)
    mse_km = vq_mse(test_x, cb_km)
    print("MSE=%.4f (%.1fs)" % (mse_km, time.time()-t0))
    results_K['K-means'] = {'mse': mse_km, 'cb': cb_km, 'history': hist_km}

    # Lloyd-Max
    cb_lloyd, _ = kmeans_train(train_x, K, iters=300, seed=42)
    mse_lloyd = vq_mse(test_x, cb_lloyd)
    results_K['Lloyd-Max'] = {'mse': mse_lloyd}
    print("  Lloyd-Max... MSE=%.4f" % mse_lloyd)

    # --- PD: K-means init, no noise ---
    print("  PD (KM init, no noise)...", end=" ", flush=True)
    cb_a, hist_a = pd_train(train_x[:5000], cb_km, predictor, K, iters=50, lr=0.03, t_match=2)
    mse_a = vq_mse(test_x, cb_a)
    print("MSE=%.4f (%+.2f%%)" % (mse_a, (mse_a-mse_km)/mse_km*100))
    results_K['PD:KM+no_noise'] = {'mse': mse_a, 'history': hist_a}

    # --- PD: K-means init, adaptive noise, cosine schedule ---
    print("  PD (KM init, adaptive, cosine)...", end=" ", flush=True)
    cb_b, hist_b = pd_train(train_x[:5000], cb_km, predictor, K, iters=50, lr=0.03, t_match=2,
                            noise_mode='adaptive', temp_start=0.5, schedule='cosine')
    mse_b = vq_mse(test_x, cb_b)
    print("MSE=%.4f (%+.2f%%)" % (mse_b, (mse_b-mse_km)/mse_km*100))
    results_K['PD:KM+adaptive_cos'] = {'mse': mse_b, 'history': hist_b}

    # --- PD: Random init, fixed Langevin, cosine schedule ---
    print("  PD (Random init, fixed Lang, cosine)...", end=" ", flush=True)
    rng_ri = np.random.RandomState(7)
    cb_rand = train_x[rng_ri.choice(len(train_x), K, replace=False)].copy()
    cb_c, hist_c = pd_train(train_x[:5000], cb_rand, predictor, K, iters=200, lr=0.05, t_match=2,
                            noise_mode='fixed', temp_start=1.0, schedule='cosine', pred_steps=5)
    mse_c = vq_mse(test_x, cb_c)
    print("MSE=%.4f (%+.2f%%)" % (mse_c, (mse_c-mse_km)/mse_km*100))
    results_K['PD:Random+fixed_cos'] = {'mse': mse_c, 'history': hist_c}

    # --- PD: Random init, adaptive noise, cosine schedule ---
    print("  PD (Random init, adaptive, cosine)...", end=" ", flush=True)
    cb_d, hist_d = pd_train(train_x[:5000], cb_rand.copy(), predictor, K, iters=200, lr=0.05, t_match=2,
                            noise_mode='adaptive', temp_start=0.5, schedule='cosine', pred_steps=5)
    mse_d = vq_mse(test_x, cb_d)
    print("MSE=%.4f (%+.2f%%)" % (mse_d, (mse_d-mse_km)/mse_km*100))
    results_K['PD:Random+adaptive_cos'] = {'mse': mse_d, 'history': hist_d}

    # K-means from random for convergence comparison
    cb_km_rand, hist_km_rand = kmeans_train(train_x, K, init=cb_rand.copy(), iters=200, seed=99, lr_max=0.3)
    results_K['K-means:from_random'] = {'history': hist_km_rand}

    all_results[K] = results_K

# ================================================================
# SUMMARY TABLE
# ================================================================
print("\n\n" + "=" * 76)
print("  COMPREHENSIVE SUMMARY")
print("=" * 76)

print("\n  %-35s" % "Method", end="")
for K in K_VALUES:
    print(" %10s" % ("K=%d" % K), end="")
print()
print("  " + "-" * 65)

methods = ['K-means', 'Lloyd-Max', 'PD:KM+no_noise', 'PD:KM+adaptive_cos',
           'PD:Random+fixed_cos', 'PD:Random+adaptive_cos']
for m in methods:
    print("  %-35s" % m, end="")
    for K in K_VALUES:
        if m in all_results[K]:
            mse = all_results[K][m]['mse']
            km_mse = all_results[K]['K-means']['mse']
            if m == 'K-means':
                print(" %10.4f" % mse, end="")
            elif m == 'Lloyd-Max':
                print(" %10.4f" % mse, end="")
            else:
                pct = (mse - km_mse) / km_mse * 100
                print(" %+9.1f%%" % pct, end="")
        else:
            print(" %10s" % "—", end="")
    print()

# Convergence trajectories
print("\n\n  Convergence from Random Init (K=64):")
K = 64
print("  %-6s %12s %12s %12s" % ("Iter", "K-means", "PD(fixed)", "PD(adapt)"))
r = all_results[K]
hist_km = r['K-means:from_random']['history']
hist_pd1 = r['PD:Random+fixed_cos']['history']
hist_pd2 = r['PD:Random+adaptive_cos']['history']

max_len = max(len(hist_km), len(hist_pd1), len(hist_pd2))
for i in range(max_len):
    vals = []
    for h in [hist_km, hist_pd1, hist_pd2]:
        vals.append(h[i][1] if i < len(h) else None)
        it_val = h[i][0] if i < len(h) else None
    if it_val is not None and (it_val <= 5 or it_val % 40 == 0 or i == max_len-1):
        s = "  %-6d" % it_val
        for v in vals:
            s += " %12.2f" % v if v is not None else " %12s" % "—"
        print(s)

# ================================================================
# Key findings
# ================================================================
print("\n\n" + "=" * 76)
print("  KEY FINDINGS")
print("=" * 76)

for K in K_VALUES:
    r = all_results[K]
    km = r['K-means']['mse']
    pd_ft = r['PD:KM+no_noise']['mse']
    pd_rand_best = min(r['PD:Random+fixed_cos']['mse'], r['PD:Random+adaptive_cos']['mse'])

    print("\n  K = %d:" % K)
    print("    K-means:        MSE = %.4f" % km)
    print("    PD fine-tune:   MSE = %.4f  (%+.2f%%)" % (pd_ft, (pd_ft-km)/km*100))
    print("    PD random init: MSE = %.4f  (%+.2f%%)" % (pd_rand_best, (pd_rand_best-km)/km*100))

    # Gradient alignment
    cb_t = torch.from_numpy(r['K-means']['cb']).float().to(device)
    data_t = torch.from_numpy(train_x[:2000]).float().to(device)
    with torch.no_grad():
        dists = torch.cdist(data_t, cb_t)
        assign = dists.argmin(dim=1)
        x_q = cb_t[assign]
        tn = torch.full((len(x_q),1), 2/T, device=device)
        eps = predictor(x_q, tn)
    gt = np.zeros((K, D_LATENT))
    gp = np.zeros((K, D_LATENT))
    assign_np = assign.cpu().numpy()
    eps_np = eps.cpu().numpy()
    for k in range(K):
        mask = assign_np == k
        if mask.sum() > 0:
            gt[k] = train_x[:2000][mask].mean(axis=0) - r['K-means']['cb'][k]
            gp[k] = eps_np[mask].mean(axis=0)
    from numpy.linalg import norm
    cos = np.dot(gp.ravel(), gt.ravel()) / (norm(gp.ravel())*norm(gt.ravel()) + 1e-15)
    print("    Gradient align: cos(PD,true) = %+.4f" % cos)

# Save
json_results = {}
for K in K_VALUES:
    json_results[str(K)] = {}
    for m, d in all_results[K].items():
        json_results[str(K)][m] = {k: float(v) if isinstance(v, (float, np.floating)) else v
                                     for k, v in d.items() if k in ['mse']}
with open(os.path.join(OUTPUT_DIR, 'pd_v10_v4_summary.json'), 'w') as f:
    json.dump(json_results, f, indent=2)
print("\n  Saved: v10_validation_output/pd_v10_v4_summary.json")
print("=" * 76)
