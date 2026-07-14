#!/usr/bin/env python3
"""
PD Full-Dim (d=2048) Codebook + Patch-Level Autoregressive Generation
======================================================================
Key difference from previous experiment:
  - d=8 (per-position): K-means near-optimal, PD no advantage
  - d=2048 (full latent): PD advantage validated (+0.07% vs K-means)

Each patch → 1 token from K=4096 codebook on full 2048-dim latent.
Image = 64 patches → 64 token sequence → GPT models spatial layout.

Pipeline:
  1. Extract latents (full 2048-dim per patch)
  2. Train PD predictor on multi-level quantized 2048-dim data
  3. K-means init + PD fine-tune → codebook K=4096
  4. Tokenize: each patch = 1 token → image = 64 tokens (8×8 grid)
  5. Train GPT on 64-token sequences (spatial autoregressive)
  6. Generate: sample 64 tokens → decode → 256×256 image

Compare: PD codebook vs K-means codebook
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_pd_full2048_ar')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')

GRID = 8  # 8×8 patches per 256×256 image
SEQ_LEN = GRID * GRID  # 64 tokens per image

print("=" * 76)
print("  PD Full-2048-dim Codebook + Patch-Level AR Generation")
print("  d=2048 | K=4096 | 64 tokens/image | GPT on spatial layout")
print("=" * 76, flush=True)

# ================================================================
# Codec models
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
LAT_CH, LAT_DS = vd['ch'], vd['ds']
D_LAT = LAT_CH * (P // LAT_DS) ** 2  # 2048
print("  d=%d" % D_LAT, flush=True)


# ================================================================
# Extract latents: full 2048-dim vectors per patch
# ================================================================
print("\n[2] Extracting latents...", flush=True)
paths = find_mini_imagenet(max_count=5000)
np.random.seed(42); np.random.shuffle(paths)
train_paths = paths[:120]
eval_paths = paths[120:128]

# Per-image: 64 patches → 64 latent vectors → 64 tokens
img_latents = []  # list of (64, 2048) tensors, one per image
img_cr = []       # list of (64, 3, 32, 32) coarse recon
img_ct = []       # list of (64, D1) coarse tokens

for path in train_paths:
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)
    patches = t_img.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)
    with torch.no_grad():
        ct = codec.coarse_encoder(patches.to(device))
        cr = codec.coarse_decoder(ct, coords)
        z = sp_enc(torch.cat([patches.to(device), cr], dim=1))  # (64, 8, 16, 16)
        z_flat = z.reshape(64, -1)  # (64, 2048)
    img_latents.append(z_flat.cpu())
    img_cr.append(cr.cpu())
    img_ct.append(ct.cpu())

n_images = len(img_latents)
all_z = torch.cat(img_latents, dim=0)  # (n_images*64, 2048)
all_z_gpu = all_z.to(device)
print("  %d images, %d latent vectors, d=%d" % (n_images, len(all_z), D_LAT), flush=True)


# ================================================================
# Train PD predictor on full 2048-dim data
# ================================================================
print("\n[3] Training PD predictor (d=%d, 12.6M params)..." % D_LAT, flush=True)

mins_z = all_z_gpu.min(dim=0)[0]
maxs_z = all_z_gpu.max(dim=0)[0]
BIT_LEVELS = [10, 8, 6, 4, 2]
T_PD = 4

def quantize_full(z, bits):
    nl = 2 ** bits
    step = (maxs_z - mins_z) / max(nl - 1, 1)
    q = torch.round((z - mins_z) / step).clamp(0, nl - 1)
    return mins_z + q * step

# Precompute quantized levels
z_quant = {}
for b in BIT_LEVELS:
    z_quant[b] = quantize_full(all_z_gpu, b).detach()

pd_pred = PDPredictor(D_LAT, hidden=1024, n_res=4).to(device)
opt_pd = torch.optim.AdamW(pd_pred.parameters(), lr=1e-3, weight_decay=1e-5)
sched_pd = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pd, T_max=100, eta_min=1e-6)

N_z = len(all_z_gpu)
for ep in range(100):
    perm = torch.randperm(N_z)
    total = 0; nb = 0
    for i in range(0, N_z, 512):
        idx = perm[i:i+512]
        z0 = all_z_gpu[idx]
        t_idx = np.random.randint(T_PD)
        bits = BIT_LEVELS[t_idx + 1]  # skip 10b (t=0)
        z_t = z_quant[bits][idx]
        tn = torch.full((len(idx), 1), t_idx / T_PD, device=device)
        loss = F.mse_loss(pd_pred(z_t, tn), z0 - z_t)
        opt_pd.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pd_pred.parameters(), 1.0)
        opt_pd.step()
        total += loss.item(); nb += 1
    sched_pd.step()
    if (ep+1) % 25 == 0:
        print("    PD predictor epoch %d: loss=%.6f" % (ep+1, total/nb), flush=True)
pd_pred.eval()


# ================================================================
# Codebook training: K=4096, d=2048
# ================================================================
print("\n[4] Training codebooks (K=4096, d=%d)..." % D_LAT, flush=True)
K_CB = 4096

def pairwise_sq_dist(A, B):
    aa = (A ** 2).sum(1)[:, None]
    bb = (B ** 2).sum(1)[None, :]
    return torch.clamp(aa - 2 * (A @ B.T) + bb, min=0)

def assign_cb(data, cb, bs=4096):
    assign = torch.zeros(len(data), dtype=torch.long, device=device)
    for i in range(0, len(data), bs):
        d = pairwise_sq_dist(data[i:i+bs], cb)
        assign[i:i+bs] = d.argmin(dim=1)
    return assign

def cb_mse(data, cb):
    a = assign_cb(data, cb)
    return F.mse_loss(cb[a], data).item()

# K-means init (30 iters — fast convergence for good start)
print("  K-means init (30 iters)...", flush=True)
torch.manual_seed(42)
init_idx = torch.randperm(N_z)[:K_CB]
cb_km = all_z_gpu[init_idx].clone()

for it in range(30):
    a = assign_cb(all_z_gpu, cb_km, bs=2048)
    for k in range(K_CB):
        mask = a == k
        if mask.sum() > 0:
            cb_km[k] = all_z_gpu[mask].mean(dim=0)
    if (it+1) % 10 == 0:
        print("    KM iter %d: MSE=%.6f" % (it+1, cb_mse(all_z_gpu[:2000], cb_km)), flush=True)

mse_km = cb_mse(all_z_gpu, cb_km)
print("  K-means final MSE: %.6f" % mse_km, flush=True)

# PD fine-tune: K-means init + PD predictor gradient
print("  PD fine-tune (30 iters)...", flush=True)
cb_pd = cb_km.clone()

for it in range(30):
    a = assign_cb(all_z_gpu, cb_pd, bs=2048)
    t_norm = torch.full((K_CB, 1), 0.0, device=device)  # t=0 (finest)
    with torch.no_grad():
        pd_correction = pd_pred(cb_pd, t_norm)  # (K, 2048)

    lr = 0.001
    for k in range(K_CB):
        mask = a == k
        if mask.sum() > 0:
            data_mean = all_z_gpu[mask].mean(dim=0)
            # Blend Lloyd-Max with PD direction
            cb_pd[k] = 0.9 * (cb_pd[k] + lr * (data_mean - cb_pd[k]) * 10) + \
                       0.1 * (cb_pd[k] + lr * pd_correction[k] * 10)
    if (it+1) % 10 == 0:
        print("    PD iter %d: MSE=%.6f" % (it+1, cb_mse(all_z_gpu[:2000], cb_pd)), flush=True)

mse_pd = cb_mse(all_z_gpu, cb_pd)
print("  PD codebook final MSE: %.6f" % mse_pd, flush=True)
print("  PD vs K-means: %+.2f%%" % ((mse_pd - mse_km) / mse_km * 100), flush=True)


# ================================================================
# Tokenize images: 64 tokens per image
# ================================================================
print("\n[5] Tokenizing images...", flush=True)

def tokenize_images(latent_list, cb):
    """Each image: (64, 2048) → (64,) token indices."""
    all_tokens = []
    for z in latent_list:
        z_gpu = z.to(device)
        d = pairwise_sq_dist(z_gpu, cb)  # (64, K)
        tokens = d.argmin(dim=1)  # (64,)
        all_tokens.append(tokens.cpu())
    return torch.stack(all_tokens)  # (n_images, 64)

def detokenize_to_patch(tokens, cb, cr):
    """tokens: (64,) → decoded image patches → 256×256 image."""
    z = cb[tokens]  # (64, 2048)
    z_4d = z.reshape(64, LAT_CH, P//LAT_DS, P//LAT_DS)
    res = sp_dec(z_4d)
    patches = (cr.to(device) + res).clamp(0, 1)
    img = patches.reshape(GRID, GRID, 3, P, P).permute(2, 0, 3, 1, 4).reshape(3, GRID*P, GRID*P)
    return img

pd_tokens = tokenize_images(img_latents, cb_pd)
km_tokens = tokenize_images(img_latents, cb_km)
print("  PD tokens: %s  KM tokens: %s" % (str(pd_tokens.shape), str(km_tokens.shape)), flush=True)

# Verify tokenization quality
def eval_tokenization_quality(tokens, cb, name):
    psnrs = []
    for i in range(min(10, len(img_latents))):
        img_rec = detokenize_to_patch(tokens[i], cb, img_cr[i])
        # Compare with original (need original images)
        path = train_paths[i]
        img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
        inp = np.array(img, dtype=np.float32) / 255.0
        rec_np = img_rec.cpu().permute(1,2,0).numpy()
        mse = np.mean((inp - rec_np)**2)
        psnrs.append(20 * math.log10(1.0/max(math.sqrt(mse), 1e-10)))
    print("  %s tokenization PSNR: %.2f dB (first 10 images)" % (name, np.mean(psnrs)), flush=True)
    return np.mean(psnrs)

psnr_pd = eval_tokenization_quality(pd_tokens, cb_pd, "PD")
psnr_km = eval_tokenization_quality(km_tokens, cb_km, "KM")


# ================================================================
# GPT: 64-token spatial sequence
# ================================================================
print("\n[6] Training GPT (64-token sequences)...", flush=True)

class CausalAttn(nn.Module):
    def __init__(self, d, h, seq):
        super().__init__()
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.h, self.dh = h, d//h
        self.register_buffer('mask', torch.tril(torch.ones(seq, seq)).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        B,S,D = x.shape
        qkv = self.qkv(x).reshape(B,S,3,self.h,self.dh).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        a = (q@k.transpose(-2,-1))/math.sqrt(self.dh)
        a = a.masked_fill(self.mask[:,:,:S,:S]==0, float('-inf'))
        a = F.softmax(a, dim=-1)
        return self.proj((a@v).transpose(1,2).reshape(B,S,D))

class GPTBlock(nn.Module):
    def __init__(self, d, h, seq):
        super().__init__()
        self.l1 = nn.LayerNorm(d); self.attn = CausalAttn(d,h,seq)
        self.l2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
    def forward(self, x):
        x = x + self.attn(self.l1(x)); x = x + self.ff(self.l2(x)); return x

class GPT(nn.Module):
    def __init__(self, vocab, d=512, h=8, layers=8, seq=SEQ_LEN+1):
        super().__init__()
        self.te = nn.Embedding(vocab, d)
        self.pe = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([GPTBlock(d,h,seq) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
    def forward(self, x):
        B,S = x.shape
        h = self.te(x) + self.pe(torch.arange(S,device=x.device).unsqueeze(0))
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))

VOCAB = K_CB + 1
BOS = K_CB

def train_gpt(tokens, label):
    model = GPT(VOCAB, d=512, h=8, layers=8).to(device)
    np_params = sum(p.numel() for p in model.parameters())
    print("  [%s] GPT: %.1fM params" % (label, np_params/1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-5)
    N = len(tokens)
    for ep in range(40):
        model.train()
        perm = torch.randperm(N)
        total = 0; nb = 0
        for i in range(0, N, 32):
            idx = perm[i:i+32]
            batch = tokens[idx].to(device)  # (B, 64)
            bos = torch.full((len(idx),1), BOS, device=device, dtype=torch.long)
            inp = torch.cat([bos, batch[:,:-1]], dim=1)  # (B, 64)
            tgt = batch  # (B, 64)
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1,VOCAB), tgt.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sch.step()
        if (ep+1)%10==0 or ep==0:
            model.eval()
            with torch.no_grad():
                si = torch.randperm(N)[:200]
                b = tokens[si].to(device)
                bos = torch.full((len(si),1),BOS,device=device,dtype=torch.long)
                inp = torch.cat([bos,b[:,:-1]],dim=1)
                acc = (model(inp).argmax(2)==b).float().mean().item()
            print("  [%s] Epoch %2d: loss=%.4f acc=%.3f" % (label, ep+1, total/nb, acc), flush=True)
    model.eval()
    return model

gpt_pd = train_gpt(pd_tokens, "PD")
gpt_km = train_gpt(km_tokens, "KM")


# ================================================================
# Generation
# ================================================================
print("\n[7] Generation...", flush=True)

@torch.no_grad()
def generate_image(model, temp=1.0, top_k=100):
    toks = torch.full((1,1), BOS, device=device, dtype=torch.long)
    for _ in range(SEQ_LEN):
        logits = model(toks)[:, -1, :] / temp
        v, _ = torch.topk(logits, min(top_k, VOCAB))
        logits[logits < v[:, -1:]] = float('-inf')
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        toks = torch.cat([toks, nxt], dim=1)
    return toks[:, 1:]


@torch.no_grad()
def reconstruct_ar(model, real_tokens, n_known):
    B = len(real_tokens)
    toks = torch.cat([torch.full((B,1),BOS,device=device,dtype=torch.long), real_tokens[:,:n_known]], dim=1)
    for pos in range(n_known, SEQ_LEN):
        nxt = model(toks)[:, -1, :].argmax(-1, keepdim=True)
        toks = torch.cat([toks, nxt], dim=1)
    return toks[:, 1:]


def save_image(tokens, cb, cr, path):
    img = detokenize_to_patch(tokens, cb, cr)
    arr = (img.detach().cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
    Image.fromarray(arr).save(path)

# Unconditional generation
print("  Unconditional generation...", flush=True)
avg_cr = torch.cat(img_cr).reshape(n_images, GRID, GRID, 3, P, P).mean(dim=(0,))  # (8,8,3,32,32)
avg_cr_flat = avg_cr.reshape(GRID*GRID, 3, P, P)

for tag, gpt, cb in [("PD", gpt_pd, cb_pd), ("KM", gpt_km, cb_km)]:
    for i in range(5):
        gen = generate_image(gpt, temp=1.0, top_k=100)
        save_image(gen[0], cb, avg_cr_flat, os.path.join(OUTPUT_DIR, 'gen_%s_%02d.png' % (tag, i)))
    print("    %s: 5 unconditional images" % tag, flush=True)

# AR reconstruction
print("  AR reconstruction...", flush=True)
results = {}
for img_idx in range(min(4, len(eval_paths))):
    path = eval_paths[img_idx]
    try: img = Image.open(path).convert('RGB').resize((256,256), Image.BILINEAR)
    except: continue
    inp = np.array(img, dtype=np.float32)/255.0
    t_img = torch.from_numpy(inp).permute(2,0,1)
    patches = t_img.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(-1,3,P,P)
    with torch.no_grad():
        ct = codec.coarse_encoder(patches.to(device))
        cr = codec.coarse_decoder(ct, coords)
        z = sp_enc(torch.cat([patches.to(device), cr], dim=1)).reshape(64, -1)

    if img_idx == 0:
        Image.fromarray((inp*255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'cmp_original.png'))

    for tag, gpt, cb in [("PD", gpt_pd, cb_pd), ("KM", gpt_km, cb_km)]:
        d = pairwise_sq_dist(z, cb)
        real_toks = d.argmin(dim=1).cpu()

        for n_known in [0, 16, 32, 48]:
            with torch.no_grad():
                ar_toks = reconstruct_ar(gpt, real_toks.unsqueeze(0).to(device), n_known)
            rec = detokenize_to_patch(ar_toks[0], cb, cr.cpu())
            rec_np = rec.cpu().permute(1,2,0).numpy()
            mse = np.mean((inp - rec_np)**2)
            psnr = 20*math.log10(1/max(math.sqrt(mse),1e-10))

            key = "%s_%d" % (tag, n_known)
            if key not in results: results[key] = []
            results[key].append(psnr)

            if img_idx == 0:
                fname = 'cmp_%s_known%d_%.1fdB.png' % (tag, n_known, psnr)
                Image.fromarray((rec_np*255).clip(0,255).astype(np.uint8)).save(
                    os.path.join(OUTPUT_DIR, fname))

    print("  img%d done" % img_idx, flush=True)


# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 76)
print("  Full-2048-dim PD Codebook + AR Generation Results")
print("=" * 76)

print("\n  Codebook MSE (d=2048, K=4096):")
print("    PD:  %.6f" % mse_pd)
print("    KM:  %.6f  (%+.2f%%)" % (mse_km, (mse_pd-mse_km)/mse_km*100))

print("\n  Tokenization PSNR (direct VQ, no AR):")
print("    PD:  %.2f dB" % psnr_pd)
print("    KM:  %.2f dB" % psnr_km)

print("\n  AR Reconstruction PSNR (%d eval images):" % min(4, len(eval_paths)))
print("  %-12s %8s %8s %8s" % ("Known", "PD-cb", "KM-cb", "PD-KM"))
print("  " + "-" * 40)
for n_k in [0, 16, 32, 48]:
    pd_v = np.mean(results.get("PD_%d" % n_k, [0]))
    km_v = np.mean(results.get("KM_%d" % n_k, [0]))
    print("  %-12s %7.2f %7.2f %+7.2f" % ("%d/64" % n_k, pd_v, km_v, pd_v - km_v))

print("\n  Output: %s" % OUTPUT_DIR)
print("=" * 76)

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump({
        'codebook_mse': {'PD': mse_pd, 'KM': mse_km},
        'tokenization_psnr': {'PD': psnr_pd, 'KM': psnr_km},
        'ar_reconstruction': {k: [float(v) for v in vals] for k, vals in results.items()},
    }, f, indent=2)
