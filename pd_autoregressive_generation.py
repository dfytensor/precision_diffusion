#!/usr/bin/env python3
"""
PD VQ Codebook + Autoregressive Image Generation
=================================================
Complete pipeline:
  1. Extract latents from v10 codec
  2. Train VQ codebook on position vectors (8-dim) using K-means
  3. Encode all patches → token sequences (256 tokens per patch, K=256 codebook)
  4. Train small GPT on token sequences (per-image = 64 patches × 256 positions = 16384 tokens)
     Actually too long. Simplify: model each PATCH's 256 positions autoregressively.
  5. Generate: sample tokens → VQ decode → ResBlockDec → image

Key design:
  - Codebook: K=256, 8-dim (one per spatial position in latent)
  - Token sequence per patch: 256 positions in raster order
  - GPT learns spatial correlations between positions
  - Generation = unconditionally sample a token sequence → decode

We compare two codebooks:
  A. K-means codebook (standard)
  B. PD-trained codebook (PD predictor gradients)

Note: For practical generation quality, we also test CONDITIONAL generation:
  Given first N tokens (from a real image), predict the rest (reconstruction via AR).
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

OUTPUT_DIR = os.path.join('F:\\precision_diffusion', 'v10_autoregressive')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')

print("=" * 76)
print("  PD VQ Codebook + Autoregressive Image Generation")
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
ch_mins = vd['ch_mins'].to(device)
ch_maxs = vd['ch_maxs'].to(device)
LAT_CH, LAT_DS = vd['ch'], vd['ds']
LAT_SP = P // LAT_DS
N_POS = LAT_SP ** 2  # 256
print("  N_POS=%d, LAT_CH=%d" % (N_POS, LAT_CH), flush=True)


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

z_all = torch.cat(all_z)       # (N_patches, 8, 16, 16)
cr_all = torch.cat(all_cr)
ct_all = torch.cat(all_ct)
print("  %d patches" % len(z_all), flush=True)

# Position vectors: (N_patches, 256, 8) → flatten to (N_patches*256, 8)
pos_vecs = z_all.permute(0, 2, 3, 1).reshape(-1, LAT_CH).cpu().numpy()
print("  Position vectors: %d x %d" % pos_vecs.shape, flush=True)


# ================================================================
# Train VQ codebook (K=256)
# ================================================================
print("\n[3] Training VQ codebook K=256...", flush=True)
K_CODEBOOK = 256
VOCAB_SIZE = K_CODEBOOK + 1  # +1 for BOS token

def kmeans_gpu(data_np, K, iters=50, seed=42):
    dt = torch.from_numpy(data_np).float().to(device)
    rng = np.random.RandomState(seed)
    cb = dt[rng.choice(len(dt), K, replace=False)].clone()
    for it in range(iters):
        assign = torch.zeros(len(dt), dtype=torch.long, device=device)
        for i in range(0, len(dt), 65536):
            dists = torch.cdist(dt[i:i+65536], cb)
            assign[i:i+65536] = dists.argmin(dim=1)
        for k in range(K):
            mask = assign == k
            if mask.sum() > 0:
                cb[k] = dt[mask].mean(dim=0)
    return cb

# Subsample for training
rng_s = np.random.RandomState(0)
pos_sub = pos_vecs[rng_s.choice(len(pos_vecs), min(80000, len(pos_vecs)), replace=False)]
codebook = kmeans_gpu(pos_sub, K_CODEBOOK, iters=50)
codebook_np = codebook.cpu().numpy()
print("  Codebook: %s, MSE=%.6f" % (codebook_np.shape, np.mean(np.min(
    np.sum((pos_sub[:, None] - codebook_np[None]) ** 2, axis=2), axis=1))), flush=True)


# ================================================================
# Tokenize: encode all patches to token maps
# ================================================================
print("\n[4] Tokenizing patches...", flush=True)
codebook_t = codebook.clone().to(device)

def tokenize_patches(z_latent):
    """z_latent: (N, 8, 16, 16) → token maps (N, 256) in raster order."""
    z_pos = z_latent.permute(0, 2, 3, 1).reshape(len(z_latent), -1, LAT_CH)  # (N, 256, 8)
    dists = torch.cdist(z_pos, codebook_t)  # (N, 256, K)
    tokens = dists.argmin(dim=2)  # (N, 256)
    return tokens

def detokenize(tokens):
    """tokens: (N, 256) → latent (N, 8, 16, 16)."""
    z_pos = codebook_t[tokens]  # (N, 256, 8)
    return z_pos.permute(0, 2, 1).reshape(len(tokens), LAT_CH, LAT_SP, LAT_SP)

with torch.no_grad():
    token_maps = tokenize_patches(z_all)  # (N_patches, 256)
    # Verify reconstruction
    z_recon = detokenize(token_maps)
    vq_mse = F.mse_loss(z_recon, z_all).item()
print("  Tokenized %d patches → %s" % (len(token_maps), str(token_maps.shape)), flush=True)
print("  VQ reconstruction MSE: %.6f" % vq_mse, flush=True)


# ================================================================
# Autoregressive Transformer (small GPT)
# ================================================================
print("\n[5] Training Autoregressive Transformer...", flush=True)

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
        out = (att @ v).transpose(1, 2).reshape(B, S, D)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class AutoregressiveModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, max_seq=N_POS + 1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, max_seq) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, tokens):
        B, S = tokens.shape
        pos = torch.arange(S, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

SEQ_LEN = N_POS  # 256
model = AutoregressiveModel(
    vocab_size=VOCAB_SIZE,
    d_model=256, n_heads=8, n_layers=6, max_seq=SEQ_LEN + 1,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print("  GPT: %d params (%.1fM), %d layers, d_model=256" % (n_params, n_params/1e6, 6), flush=True)

# Training data: token_maps → (N_patches, 256)
# Add BOS token at position 0, shift targets
BOS = K_CODEBOOK
train_tokens = token_maps.cpu().numpy()  # (N_patches, 256)

# Prepare: input = [BOS, t0, t1, ..., t254], target = [t0, t1, ..., t255]
# That's 256 length input → predict 256 tokens
N_train = len(train_tokens)
print("  Training on %d sequences of length %d" % (N_train, SEQ_LEN), flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
bs = 128

for ep in range(50):
    model.train()
    perm = torch.randperm(N_train)
    total_loss = 0; nb = 0
    for i in range(0, N_train, bs):
        idx = perm[i:i+bs]
        batch = torch.from_numpy(train_tokens[idx]).to(device)  # (B, 256)
        # Prepend BOS
        bos_col = torch.full((len(idx), 1), BOS, device=device, dtype=torch.long)
        inp = torch.cat([bos_col, batch[:, :-1]], dim=1)  # (B, 256) = [BOS, t0..t253]
        tgt = batch  # (B, 256) = [t0, t1, ..., t255]

        logits = model(inp)  # (B, 256, vocab)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); nb += 1
    sched.step()
    if (ep+1) % 10 == 0 or ep == 0:
        # Compute accuracy
        model.eval()
        with torch.no_grad():
            sample_idx = torch.randperm(N_train)[:500]
            batch = torch.from_numpy(train_tokens[sample_idx]).to(device)
            bos_col = torch.full((500, 1), BOS, device=device, dtype=torch.long)
            inp = torch.cat([bos_col, batch[:, :-1]], dim=1)
            logits = model(inp)
            pred = logits.argmax(dim=2)
            acc = (pred == batch).float().mean().item()
        print("  Epoch %2d: loss=%.4f  acc=%.3f" % (ep+1, total_loss/nb, acc), flush=True)

model.eval()
print("  Done.", flush=True)


# ================================================================
# Generation: autoregressive sampling
# ================================================================
print("\n[6] Autoregressive generation...", flush=True)

@torch.no_grad()
def generate_tokens(model, n_samples=1, temperature=1.0, top_k=None):
    """Generate token sequences autoregressively."""
    model.eval()
    tokens = torch.full((n_samples, 1), BOS, device=device, dtype=torch.long)

    for pos in range(SEQ_LEN):
        logits = model(tokens)  # (B, pos+1, vocab)
        logits = logits[:, -1, :] / temperature  # (B, vocab)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, VOCAB_SIZE))
            logits[logits < v[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)  # (B, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens[:, 1:]  # remove BOS, (B, 256)


@torch.no_grad()
def reconstruct_tokens(model, real_tokens, mask_pos):
    """Given first mask_pos tokens, reconstruct the rest autoregressively.
    Tests how well the model reconstructs known images."""
    model.eval()
    B = real_tokens.shape[0]
    tokens = real_tokens[:, :mask_pos].clone()
    # Prepend BOS
    bos = torch.full((B, 1), BOS, device=device, dtype=torch.long)
    tokens = torch.cat([bos, tokens], dim=1)

    for pos in range(mask_pos, SEQ_LEN):
        logits = model(tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens[:, 1:]  # remove BOS


def tokens_to_image(tokens, coarse_recon):
    """Decode tokens → latent → residual → final image patch."""
    z_dq = detokenize(tokens)  # (N, 8, 16, 16)
    res = sp_dec(z_dq)
    return (coarse_recon + res).clamp(0, 1)


# ================================================================
# Experiment A: Unconditional generation
# ================================================================
print("\n  Experiment A: Unconditional generation (10 samples)...", flush=True)
for i in range(10):
    gen_tokens = generate_tokens(model, n_samples=64, temperature=1.0, top_k=40)  # (64, 256)

    # Need coarse recon for 64 patches (8x8 grid = 256x256 image)
    # Use average coarse token for unconditional generation
    avg_ct = ct_all.mean(dim=0, keepdim=True).expand(64, -1)
    with torch.no_grad():
        cr_gen = codec.coarse_decoder(avg_ct, coords)

    patches_gen = tokens_to_image(gen_tokens, cr_gen)
    # Reshape to image (8x8 patches of 32x32)
    img_gen = patches_gen.reshape(8, 8, 3, P, P).permute(2, 0, 3, 1, 4).reshape(3, 256, 256)
    img_np = img_gen.detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((img_np * 255).clip(0, 255).astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, 'gen_uncond_%02d.png' % i))

print("  Saved 10 unconditional samples", flush=True)


# ================================================================
# Experiment B: Conditional reconstruction (masked AR)
# ================================================================
print("\n  Experiment B: AR reconstruction (given first N tokens)...", flush=True)

# Use eval images
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
        real_tokens = tokenize_patches(z)

    # Original
    Image.fromarray((inp*255).astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, 'rec%d_00original.png' % img_idx))

    # Full VQ reconstruction (no AR)
    with torch.no_grad():
        full_recon = tokens_to_image(real_tokens, cr)
    full_img = full_recon.reshape(nH, nW, 3, P, P).permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]
    mse = np.mean((inp - full_img.cpu().permute(1,2,0).numpy())**2)
    psnr_full = 20 * math.log10(1.0 / max(math.sqrt(mse), 1e-10))
    Image.fromarray((full_img.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, 'rec%d_01vqfull_%.1fdB.png' % (img_idx, psnr_full)))

    # AR reconstruction with different mask ratios
    for mask_frac in [0.0, 0.25, 0.5, 0.75]:
        mask_pos = int(SEQ_LEN * mask_frac)
        label = "first%d%%" % int(mask_frac * 100)

        with torch.no_grad():
            ar_tokens = reconstruct_tokens(model, real_tokens, mask_pos)
            ar_recon = tokens_to_image(ar_tokens, cr)

        ar_img = ar_recon.reshape(nH, nW, 3, P, P).permute(2, 0, 3, 1, 4).reshape(3, nH*P, nW*P)[:, :H, :W]
        mse = np.mean((inp - ar_img.cpu().permute(1,2,0).numpy())**2)
        psnr = 20 * math.log10(1.0 / max(math.sqrt(mse), 1e-10))

        tag = "ar_%s_%.1fdB" % (label, psnr)
        Image.fromarray((ar_img.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, 'rec%d_%s.png' % (img_idx, tag)))
        print("    img%d %s: PSNR=%.2f dB" % (img_idx, label, psnr), flush=True)


# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 76)
print("  Autoregressive Experiment Summary")
print("=" * 76)

print("""
  Codebook: K=256, 8-dim position vectors
  Token sequence: 256 per patch (raster order in 16x16 latent grid)
  GPT: 6 layers, d_model=256, 8 heads, ~3M params
  Training: 50 epochs on %d patches

  Experiment A: 10 unconditional samples saved
  Experiment B: AR reconstruction with 0%%/25%%/50%%/75%% context

  Output: %s
""" % (N_train, OUTPUT_DIR))
print("=" * 76)
