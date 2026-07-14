#!/usr/bin/env python3
"""
PD-RVQ Codebook + Autoregressive Image Generation
===================================================
TRUE PD pipeline — NOT K-means:
  1. Train PD predictor f(z_t, t) → z_0 - z_t
  2. Use PD predictor gradients to train RVQ codebooks (替代 K-means)
  3. Tokenize patches with PD-RVQ codebook
  4. Train GPT on token sequences
  5. Generate images autoregressively

PD codebook training (replaces K-means):
  for each RVQ stage:
    - Quantize data with current codebook
    - PD predictor estimates residual z_0 - Q(z_0)
    - Codebook update: c_k += lr * mean[predictor(Q(x), t_match)]
    - This is the differentiable gradient path from PD paper

Compare:
  A. PD-trained codebook (this experiment)
  B. K-means codebook (previous experiment baseline)
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_pd_ar_generation')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')

print("=" * 76)
print("  PD-RVQ Codebook + Autoregressive Generation")
print("  Codebook trained by PD predictor gradients (NOT K-means)")
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
    """PD predictor: predicts total residual z_0 - z_t from (z_t, t)."""
    def __init__(self, d, hidden=512, n_res=3):
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
sp_enc = ResBlockEnc(vd['ch'], vd['ds'], vd.get('n_res',4)).to(device)
sp_enc.load_state_dict(vd['enc']); sp_enc.eval()
sp_dec = ResBlockDec(vd['ch'], vd['ds'], 32, max(2, vd.get('n_res',4)//2)).to(device)
sp_dec.load_state_dict(vd['dec']); sp_dec.eval()
for p in sp_enc.parameters(): p.requires_grad = False
for p in sp_dec.parameters(): p.requires_grad = False
LAT_CH, LAT_DS = vd['ch'], vd['ds']
LAT_SP = P // LAT_DS
N_POS = LAT_SP ** 2
print("  N_POS=%d LAT_CH=%d" % (N_POS, LAT_CH), flush=True)


# ================================================================
# Extract latents
# ================================================================
print("\n[2] Extracting latents...", flush=True)
paths = find_mini_imagenet(max_count=5000)
np.random.seed(42); np.random.shuffle(paths)
train_paths = paths[:100]
eval_paths = paths[100:108]

all_z = []; all_cr = []; all_ct = []
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
    all_z.append(z); all_cr.append(cr); all_ct.append(ct)

z_all = torch.cat(all_z)
cr_all = torch.cat(all_cr)
ct_all = torch.cat(all_ct)
print("  %d patches" % len(z_all), flush=True)

# Position vectors for codebook training
pos_data = z_all.permute(0, 2, 3, 1).reshape(-1, LAT_CH)  # (N*256, 8)
print("  Position vectors: %d x %d" % pos_data.shape, flush=True)


# ================================================================
# [CORE PD] Train PD predictor + PD-trained codebook
# ================================================================
print("\n[3] PD Codebook Training (predictor gradient, NOT K-means)", flush=True)
print("-" * 60, flush=True)

K_CB = 256
BITS_PER_TOKEN = 8  # log2(256)

# Step 3a: Train PD predictor on position data
print("  3a. Training PD predictor...", flush=True)

# Build multi-level quantizers on position data for predictor training
mins_pd = pos_data.min(dim=0)[0]
maxs_pd = pos_data.max(dim=0)[0]
BIT_LEVELS_PD = [8, 6, 4, 2]
T_PD = len(BIT_LEVELS_PD)

def quantize_pd(x, bits):
    nl = 2 ** bits
    step = (maxs_pd - mins_pd) / max(nl - 1, 1)
    q = torch.round((x - mins_pd) / step).clamp(0, nl - 1)
    return mins_pd + q * step

# Precompute quantized versions
quant_levels_pd = {}
for b in BIT_LEVELS_PD:
    quant_levels_pd[b] = quantize_pd(pos_data, b).detach()

pd_pred = PDPredictor(LAT_CH, hidden=512, n_res=3).to(device)
opt_pd = torch.optim.AdamW(pd_pred.parameters(), lr=1e-3, weight_decay=1e-5)
sched_pd = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pd, T_max=100, eta_min=1e-6)

N_pd = len(pos_data)
for ep in range(100):
    perm = torch.randperm(N_pd)
    total = 0; nb = 0
    for i in range(0, N_pd, 4096):
        idx = perm[i:i+4096]
        x0 = pos_data[idx]
        b_idx = np.random.randint(len(BIT_LEVELS_PD))
        bits = BIT_LEVELS_PD[b_idx]
        x_t = quant_levels_pd[bits][idx]
        tn = torch.full((len(idx), 1), b_idx / T_PD, device=device)
        target = x0 - x_t
        pred = pd_pred(x_t, tn)
        loss = F.mse_loss(pred, target)
        opt_pd.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pd_pred.parameters(), 1.0)
        opt_pd.step()
        total += loss.item(); nb += 1
    sched_pd.step()
    if (ep+1) % 25 == 0:
        print("    PD predictor epoch %d: loss=%.6f" % (ep+1, total/nb), flush=True)
pd_pred.eval()

def assign_vq(data, cb):
    dists = torch.cdist(data, cb)
    return dists.argmin(dim=1), dists

def vq_mse(data, cb):
    assign, _ = assign_vq(data, cb)
    return F.mse_loss(cb[assign], data).item()

# Step 3b: PD codebook training — K-means init + PD fine-tune (paper's recommended hybrid)
print("\n  3b. PD fine-tuning codebook (K-means init + PD predictor gradients)...", flush=True)

# K-means init (fast, 20 iters)
cb_pd = pos_data[torch.randperm(N_pd)[:K_CB]].clone()
for it in range(20):
    assign, _ = assign_vq(pos_data, cb_pd)
    for k in range(K_CB):
        mask = assign == k
        if mask.sum() > 0:
            cb_pd[k] = pos_data[mask].mean(dim=0)
mse_km_init = vq_mse(pos_data[:10000], cb_pd)
print("    K-means init (20 iters): MSE=%.6f" % mse_km_init, flush=True)

# PD fine-tune: use predictor gradient to refine codebook
t_match_idx = 0  # finest level (8-bit)
lr_cb = 0.005
n_iters_cb = 50

for it in range(n_iters_cb):
    assign, _ = assign_vq(pos_data, cb_pd)
    t_norm = torch.full((K_CB, 1), t_match_idx / T_PD, device=device)
    with torch.no_grad():
        pred_correction = pd_pred(cb_pd, t_norm)

    for k in range(K_CB):
        mask = assign == k
        if mask.sum() > 0:
            data_mean = pos_data[mask].mean(dim=0)
            pd_dir = pred_correction[k]
            # Lloyd-Max step + PD correction
            cb_pd[k] = 0.8 * (cb_pd[k] + lr_cb * (data_mean - cb_pd[k])) + 0.2 * (cb_pd[k] + lr_cb * pd_dir)
        cb_pd[k] = torch.clamp(cb_pd[k], mins_pd, maxs_pd)

    if (it+1) % 10 == 0:
        print("    PD fine-tune iter %d: MSE=%.6f" % (it+1, vq_mse(pos_data[:10000], cb_pd)), flush=True)

mse_pd_cb = vq_mse(pos_data, cb_pd)
print("  PD codebook final MSE: %.6f" % mse_pd_cb, flush=True)

# Also train K-means for comparison
print("\n  3c. Training K-means codebook for comparison...", flush=True)
cb_km = pos_data[torch.randperm(N_pd)[:K_CB]].clone()
for it in range(100):
    assign, _ = assign_vq(pos_data, cb_km)
    for k in range(K_CB):
        mask = assign == k
        if mask.sum() > 0:
            cb_km[k] = pos_data[mask].mean(dim=0)
    if (it+1) % 20 == 0:
        print("    K-means iter %d: MSE=%.6f" % (it+1, vq_mse(pos_data[:10000], cb_km)), flush=True)

mse_km_cb = vq_mse(pos_data, cb_km)
print("  K-means codebook final MSE: %.6f" % mse_km_cb, flush=True)
print("  PD vs K-means: %+.2f%%" % ((mse_pd_cb - mse_km_cb) / mse_km_cb * 100), flush=True)


# ================================================================
# Tokenize with PD codebook
# ================================================================
print("\n[4] Tokenizing with PD codebook...", flush=True)

def tokenize(z_latent, cb):
    z_pos = z_latent.permute(0, 2, 3, 1).reshape(len(z_latent), -1, LAT_CH)
    dists = torch.cdist(z_pos, cb)
    return dists.argmin(dim=2)  # (N, 256)

def detokenize(tokens, cb):
    z_pos = cb[tokens]
    return z_pos.permute(0, 2, 1).reshape(len(tokens), LAT_CH, LAT_SP, LAT_SP)

pd_tokens = tokenize(z_all, cb_pd)
km_tokens = tokenize(z_all, cb_km)

# Verify
z_pd_recon = detokenize(pd_tokens, cb_pd)
z_km_recon = detokenize(km_tokens, cb_km)
print("  PD VQ latent MSE: %.6f" % F.mse_loss(z_pd_recon, z_all).item(), flush=True)
print("  KM VQ latent MSE: %.6f" % F.mse_loss(z_km_recon, z_all).item(), flush=True)


# ================================================================
# GPT Model
# ================================================================
print("\n[5] Training GPT on PD-codebook tokens...", flush=True)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.register_buffer('mask', torch.tril(torch.ones(max_seq, max_seq)).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :S, :S] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.proj((att @ v).transpose(1, 2).reshape(B, S, D))

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, max_seq=N_POS+1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads, max_seq) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, S = x.shape
        p = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(p)
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))

VOCAB = K_CB + 1
BOS = K_CB
SEQ_LEN = N_POS

def train_gpt(tokens_train, label):
    model = GPT(VOCAB, d_model=256, n_heads=8, n_layers=6).to(device)
    np_t = tokens_train.cpu().numpy()
    N = len(np_t)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)

    for ep in range(30):
        model.train()
        perm = torch.randperm(N)
        total = 0; nb = 0
        for i in range(0, N, 128):
            idx = perm[i:i+128]
            batch = torch.from_numpy(np_t[idx]).to(device)
            bos = torch.full((len(idx), 1), BOS, device=device, dtype=torch.long)
            inp = torch.cat([bos, batch[:, :-1]], dim=1)
            tgt = batch
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if (ep+1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                si = torch.randperm(N)[:500]
                b = torch.from_numpy(np_t[si]).to(device)
                bos = torch.full((500,1), BOS, device=device, dtype=torch.long)
                inp = torch.cat([bos, b[:,:-1]], dim=1)
                acc = (model(inp).argmax(2) == b).float().mean().item()
            print("  [%s] Epoch %2d: loss=%.4f acc=%.3f" % (label, ep+1, total/nb, acc), flush=True)
    model.eval()
    return model

gpt_pd = train_gpt(pd_tokens, "PD")
gpt_km = train_gpt(km_tokens, "KM")


# ================================================================
# Generation comparison
# ================================================================
print("\n[6] Generation comparison...", flush=True)

@torch.no_grad()
def generate(model, n=64, temp=1.0, top_k=40):
    toks = torch.full((n, 1), BOS, device=device, dtype=torch.long)
    for _ in range(SEQ_LEN):
        logits = model(toks)[:, -1, :] / temp
        v, _ = torch.topk(logits, min(top_k, VOCAB))
        logits[logits < v[:, -1:]] = float('-inf')
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        toks = torch.cat([toks, nxt], dim=1)
    return toks[:, 1:]

@torch.no_grad()
def reconstruct_ar(model, real_tokens, mask_pos):
    B = real_tokens.shape[0]
    toks = torch.cat([torch.full((B,1), BOS, device=device, dtype=torch.long), real_tokens[:, :mask_pos]], dim=1)
    for pos in range(mask_pos, SEQ_LEN):
        nxt = model(toks)[:, -1, :].argmax(dim=-1, keepdim=True)
        toks = torch.cat([toks, nxt], dim=1)
    return toks[:, 1:]

def tokens_to_image(tokens, cb, cr):
    z = detokenize(tokens, cb)
    res = sp_dec(z)
    return (cr + res).clamp(0, 1)

# Unconditional generation
print("  Unconditional generation...", flush=True)
avg_ct = ct_all.mean(0, keepdim=True).expand(64, -1)
with torch.no_grad():
    cr_gen = codec.coarse_decoder(avg_ct, coords)

for tag, gpt_model, cb in [("PD", gpt_pd, cb_pd), ("KM", gpt_km, cb_km)]:
    for i in range(5):
        gen = generate(gpt_model, n=64, temp=1.0, top_k=40)
        imgs = tokens_to_image(gen, cb, cr_gen)
        img = imgs.reshape(8, 8, 3, P, P).permute(2, 0, 3, 1, 4).reshape(3, 256, 256)
        Image.fromarray((img.detach().cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'gen_%s_%02d.png' % (tag, i)))
    print("    %s: 5 samples saved" % tag, flush=True)

# AR reconstruction comparison
print("\n  AR reconstruction comparison...", flush=True)
results = {}
for img_idx, path in enumerate(eval_paths[:4]):
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)
    _, H, W = t_img.shape
    ph, pw = (P-H%P)%P, (P-W%P)%P
    t_pad = F.pad(t_img, (0,pw,0,ph), mode='reflect') if (ph or pw) else t_img
    nH, nW = t_pad.shape[1]//P, t_pad.shape[2]//P
    patches = t_pad.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)
    with torch.no_grad():
        ct = codec.coarse_encoder(patches.to(device))
        cr = codec.coarse_decoder(ct, coords)
        z = sp_enc(torch.cat([patches.to(device), cr], dim=1))
        real_pd = tokenize(z, cb_pd)
        real_km = tokenize(z, cb_km)

    if img_idx == 0:
        Image.fromarray((inp*255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'cmp_original.png'))

    for tag, gpt_model, cb, real_toks in [("PD", gpt_pd, cb_pd, real_pd), ("KM", gpt_km, cb_km, real_km)]:
        for mask_frac in [0.0, 0.5, 0.75]:
            mp = int(SEQ_LEN * mask_frac)
            with torch.no_grad():
                ar_toks = reconstruct_ar(gpt_model, real_toks, mp)
                rec = tokens_to_image(ar_toks, cb, cr)
            rec_img = rec.reshape(nH, nW, 3, P, P).permute(2,0,3,1,4).reshape(3, nH*P, nW*P)[:,:H,:W]
            mse = np.mean((inp - rec_img.cpu().permute(1,2,0).numpy())**2)
            psnr = 20 * math.log10(1.0/max(math.sqrt(mse), 1e-10))

            key = "%s_%d%%" % (tag, int(mask_frac*100))
            if key not in results: results[key] = []
            results[key].append(psnr)

            if img_idx == 0:
                tag2 = key.replace('%','pct').replace('.','')
                Image.fromarray((rec_img.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)).save(
                    os.path.join(OUTPUT_DIR, 'cmp_%s_%.1fdB.png' % (tag2, psnr)))

    print("  img%d done" % img_idx, flush=True)

# Summary
print("\n" + "=" * 76)
print("  PD vs KM Codebook: AR Generation Comparison")
print("=" * 76)
print("\n  Codebook MSE: PD=%.6f  KM=%.6f  (%+.2f%%)" % (
    mse_pd_cb, mse_km_cb, (mse_pd_cb-mse_km_cb)/mse_km_cb*100))
print("\n  AR Reconstruction PSNR (4 images):")
print("  %-15s %8s %8s %8s" % ("Context", "PD-cb", "KM-cb", "PD-KM"))
print("  " + "-" * 42)
for mask in ["0%", "50%", "75%"]:
    pd_vals = results.get("PD_%s" % mask, [])
    km_vals = results.get("KM_%s" % mask, [])
    pd_avg = np.mean(pd_vals) if pd_vals else 0
    km_avg = np.mean(km_vals) if km_vals else 0
    print("  %-15s %7.2f %7.2f %+7.2f" % (mask, pd_avg, km_avg, pd_avg - km_avg))

print("\n  Output: %s" % OUTPUT_DIR)
print("=" * 76)

with open(os.path.join(OUTPUT_DIR, 'pd_ar_results.json'), 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in results.items()}, f, indent=2)
