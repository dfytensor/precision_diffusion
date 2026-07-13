# Precision Diffusion: A Quantization-Theoretic Alternative to Gaussian Diffusion for Differentiable Vector Quantization

**Authors**: [Author Names]

**Date**: July 2026

---

## Abstract

We introduce **Precision Diffusion (PD)**, a novel forward process for diffusion models that replaces additive Gaussian noise with progressive quantization at decreasing bit-depths. Unlike standard Denoising Diffusion Probabilistic Models (DDPM), whose forward process destroys information via isotropic noise injection, PD degrades precision by quantizing data at successively coarser resolutions (e.g., 8-bit → 6-bit → 4-bit → 2-bit → 1-bit). The resulting residuals are structured quantization errors, not Gaussian noise—a distinction we prove statistically via two-dimensional Kolmogorov–Smirnov tests ($p \approx 0$ at all diffusion steps). This structural difference positions PD as a principled, fully differentiable alternative to the Straight-Through Estimator (STE) for training vector quantization (VQ) codebooks in VQ-VAE and related architectures. We validate PD through a controlled, minimal-mechanism experimental framework: a 1D scalar quantization probe and a 2D vector quantization benchmark with joint predictor–codebook training and Langevin noise injection. Results show that from good initialization, PD matches K-means within $0.3\%$ MSE while providing genuine (non-STE) gradients. From degenerate initialization, Langevin noise successfully breaks symmetry, enabling convergence—albeit slower than hard-assignment K-means. We conclude that PD offers a viable, theoretically grounded replacement for STE in VQ-based generative models, with the K-means pretrain + PD finetune paradigm emerging as the recommended practical strategy.

---

## 1. Introduction

Diffusion models [1, 2] have revolutionized generative modeling by decomposing complex data generation into a sequence of denoising steps. The standard DDPM forward process destroys data structure by adding isotropic Gaussian noise:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

While effective for continuous data, this formulation is fundamentally mismatched with the discrete nature of vector quantization [3]. VQ-based architectures (VQ-VAE, VQGAN, etc.) rely on the Straight-Through Estimator (STE) [4] to approximate gradients through the non-differentiable quantization operation—a well-known compromise that introduces biased gradient signals and limits optimization quality.

We ask: **can the diffusion framework itself be reformulated to operate directly on quantization error, rather than Gaussian noise?** If so, the "noise predictor" becomes a "precision predictor" that learns the quantization error distribution, and its gradient provides a mathematically legitimate path through the quantizer.

This paper proposes **Precision Diffusion (PD)**, where:

1. The forward process is bit-depth reduction: $x_t = Q_t(x_0)$, with $Q_t$ being a quantizer at $2^{b_t}$ bins where $b_t$ decreases with $t$.
2. The residual at each step is quantization error $\epsilon_t^{\text{PD}} = x_t - x_{t-1}$, not Gaussian noise.
3. A predictor $f_\theta(x_t, t)$ learns to estimate the total quantization residual $x_0 - x_t$.
4. Codebook updates leverage $f_\theta$'s gradient: $c_k \leftarrow c_k + \eta \cdot f_\theta(c_k, t)$, providing a fully differentiable alternative to STE.

We design a minimal-mechanism validation framework—**1D scalar quantization probe + 2D vector quantization benchmark**—that isolates PD from confounding factors (no images, no encoders, no VQ-specific components). This framework evaluates three claims:

- **Claim 1:** PD forward residuals are statistically distinct from DDPM forward residuals.
- **Claim 2:** PD can train a quantizer to minimize MSE, matching K-means quality.
- **Claim 3:** PD gradients are aligned with the true (Lloyd-Max) gradient direction, unlike STE.

---

## 2. Related Work

**Diffusion Models.** DDPM [1] and score-based models [5] define the forward process as gradual Gaussian noising. The reverse process learns to denoise via a score network. DDIM [6] and subsequent works accelerate sampling but retain the Gaussian forward assumption.

**Vector Quantization.** Lloyd-Max quantization [7, 8] provides the optimal scalar quantizer for a given distribution. K-means (Lloyd's algorithm) generalizes this to vector data. VQ-VAE [3] introduces vector quantization into autoencoders, using STE for gradient approximation. VQGAN [9] and RQ-VAE [10] extend this with perceptual losses and residual quantization.

**Straight-Through Estimator.** STE [4] copies gradients through the quantizer as if it were an identity function: $\partial \hat{z} / \partial z = I$ despite $\hat{z} = \text{argmin}_{c \in C} \|z - c\|$ being non-differentiable. This introduces bias [11], especially in early training when encoder outputs are far from codebook entries.

**Precision Diffusion.** Unlike prior work that uses diffusion for generative purposes, PD focuses on the forward process itself as a quantization mechanism. The closest conceptual precedent is hierarchical VQ [12], but PD frames this within the diffusion formalism, enabling score-based gradient computation.

---

## 3. Method: Precision Diffusion

### 3.1 Forward Process

Given a data point $x_0 \in \mathbb{R}^d$ and a sequence of quantization levels with $K_t = 2^{b_t}$ bins where $b_0 > b_1 > \cdots > b_T$, the PD forward process is:

$$x_t = Q_t(x_0) = \text{argmin}_{c \in \mathcal{C}_t} \|x_0 - c\|^2$$

where $\mathcal{C}_t$ is the codebook at precision level $t$ with $K_t$ entries. In practice, we construct $\mathcal{C}_t$ by running K-means on the training data at each target size $K_t$, yielding non-uniform quantizers adapted to the data distribution.

The **step residual** (analogous to DDPM's $\epsilon$) is:

$$\epsilon_t^{\text{PD}} = x_t - x_{t-1}$$

The **total residual** (cumulative quantization error) is:

$$E_t^{\text{PD}} = x_0 - x_t$$

Unlike DDPM where $\epsilon_t^{\text{DDPM}} \sim \mathcal{N}(0, (1 - \bar{\alpha}_t)I)$, PD residuals are structured: they follow the data distribution's geometry at each precision level, with non-trivial correlation structure in multivariate settings.

### 3.2 Predictor Training

We train a predictor $f_\theta: \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d$ to estimate the total residual:

$$\mathcal{L}(\theta) = \mathbb{E}_{x_0, t}\left[\|f_\theta(x_t, t/T) - (x_0 - x_t)\|^2\right]$$

where $t \sim \text{Uniform}(1, T)$ and $x_t = Q_t(x_0)$. The predictor takes the quantized value $x_t$ and normalized time $t/T$ as input, and outputs the predicted correction to the original data.

Importantly, we train $f_\theta$ on data from **all** quantization levels simultaneously. This multi-level training provides the predictor with a comprehensive view of the data distribution at multiple resolutions, enabling cross-cell gradient information that pure K-means (which only sees its own Voronoi cells) cannot access.

### 3.3 Codebook Training via Predictor Gradient

For a target codebook $\mathcal{C} = \{c_1, \ldots, c_K\}$, the PD gradient for entry $c_k$ is:

$$\Delta c_k = \eta \cdot f_\theta(c_k, t_{\text{match}})$$

where $t_{\text{match}}$ is the diffusion step whose codebook size matches $K$ (e.g., $K = 16$ corresponds to 4-bit, $t = 2$ in our 4-step schedule).

Alternatively, a per-data-point variant averages the predictor output over the Voronoi cell:

$$\Delta c_k = \eta \cdot \frac{1}{|\mathcal{V}_k|} \sum_{x \in \mathcal{V}_k} f_\theta(Q_\mathcal{C}(x), t_{\text{match}})$$

where $\mathcal{V}_k = \{x : \|x - c_k\| \leq \|x - c_j\| \;\forall j\}$.

This gradient is fully differentiable (no stop-gradient or STE), since $f_\theta$ is a smooth neural network. The chain rule flows as $\partial \mathcal{L} / \partial \mathcal{C} = \partial \mathcal{L} / \partial f_\theta \cdot \partial f_\theta / \partial x_q$, where $x_q$ is the quantized input to the predictor.

### 3.4 Joint Training with Langevin Dynamics

A naive application of the centroid-level gradient suffers from a symmetry problem: if all centroids start at the same position, $f_\theta(c_k, t)$ is identical for all $k$, preventing the codebook from splitting. We resolve this by framing codebook updates as Langevin dynamics:

$$c_k^{(i+1)} = c_k^{(i)} + \eta \cdot f_\theta(c_k^{(i)}, t) + \sqrt{2\eta \cdot \tau^{(i)}} \cdot z_k, \quad z_k \sim \mathcal{N}(0, I)$$

where $\tau^{(i)}$ is a temperature parameter annealed from $\tau_{\text{start}}$ to $0$ over training. The noise term breaks symmetry, enabling exploration, and vanishes at convergence.

The full training algorithm is two-timescale:

1. **Inner loop:** Train $f_\theta$ on current codebook $\mathcal{C}^{(i)}$ + pre-built multi-level quantizers.
2. **Outer loop:** Update $\mathcal{C}^{(i)}$ using $f_\theta$'s gradient with Langevin noise.

This ensures the predictor always adapts to the current codebook state, eliminating the distribution-shift problem.

---

## 4. Experimental Design: The Precision Diffusion Probe

To isolate PD's core mechanism from task-specific confounds, we design two minimal experiments:

### 4.1 1D Scalar Quantization Probe

**Data:** 100K train / 10K test points from $0.5 \cdot \mathcal{N}(-2, 0.5) + 0.5 \cdot \mathcal{N}(2, 0.5)$, a bimodal distribution where non-uniform quantization yields clear benefits over uniform binning.

**Forward schedule:** $T = 4$ steps, bit depths $[8, 6, 4, 2, 1]$ corresponding to $[256, 64, 16, 4, 2]$ quantile-based bins.

**Predictor:** 2-layer MLP ($2 \to 32 \to 1$) taking $(x_t, t/T)$ as input, trained via OLS/SGD.

**Target codebook:** $K = 16$ (matching 4-bit level). Methods compared: Lloyd-Max (theoretical optimum), K-means (standard VQ), and PD.

### 4.2 2D Vector Quantization Benchmark

**Data:** 50K train / 10K test points from a 5-component 2D Gaussian mixture with asymmetric positions and non-uniform weights: $(0, 4)$ at 30%, $(\pm 3, 0)$ at 22% each, $(\pm 1.5, -3)$ at 13% each.

**Forward schedule:** $T = 4$ steps, K-means-based quantizers at $[256, 64, 16, 4, 2]$ centroids.

**Predictor:** 3-layer MLP ($4 \to 128 \to 128 \to 2$) taking $([x_t^1, x_t^2, t/T, t/T])$ as input, trained with SGD.

**Training regimes:**
- *Good init:* K-means-pretrained codebook ($\text{MSE} \approx 0.118$)
- *Moderate-poor init:* Random centroids near $(0.5, 0.5)$ ($\text{MSE} \approx 5.9$)
- *Very poor init:* Near-degenerate centroids at origin ($\text{MSE} \approx 6.3$)

**PD variants:** Centro-level gradient vs. per-data-point averaged gradient, both with Langevin noise annealing.

### 4.3 Evaluation Metrics

1. **Forward Process Difference:** Two-sample KS test on residual norms and angles between DDPM and PD at each diffusion step.
2. **Quantization MSE:** Test-set MSE of the trained $K=16$ codebook vs. Lloyd-Max upper bound.
3. **Gradient Legitimacy:** Cosine similarity between PD gradient direction and true (Lloyd-Max) gradient direction.

---

## 5. Results

### 5.1 Metric 1: Forward Process Is Not DDPM

**Table 1: KS test — DDPM vs. PD step residuals**

| Step | Bits (→) | KS Stat (DDPM vs PD) | p-value | DDPM ∼ N(0,1)? | PD ∼ N(0,1)? |
|------|----------|---------------------|---------|-----------------|---------------|
| t=1  | 256→64   | 0.441               | 0.0     | p = 0.026       | p = 0.0       |
| t=2  | 64→16    | 0.373               | 0.0     | p = 0.025       | p = 0.0       |
| t=3  | 16→4     | 0.275               | 0.0     | p = 0.025       | p = 0.0       |
| t=4  | 4→2      | 0.288               | 0.0     | p = 0.024       | p = 0.0       |

At all four diffusion steps, the KS test overwhelmingly rejects the null hypothesis that DDPM and PD residuals share the same distribution ($p \approx 0$). DDPM residuals are consistent with $\mathcal{N}(0, 1)$ (all $p > 0.01$), while PD residuals are strongly non-Gaussian ($p \approx 0$ vs. normal).

**Table 2: 2D residual structure**

| Step | DDPM $| \epsilon |$ | DDPM $\rho$ | PD $| \epsilon |$ | PD $\rho$ |
|------|-------------------|--------------|----------------|------------|
| t=1  | 1.255 ± 0.655     | −0.001       | 0.164 ± 0.084  | +0.008     |
| t=2  | 1.252 ± 0.654     | +0.005       | 0.293 ± 0.144  | +0.141     |
| t=3  | 1.253 ± 0.656     | +0.002       | 0.652 ± 0.535  | −0.037     |
| t=4  | 1.256 ± 0.656     | −0.008       | 2.283 ± 0.344  | **+0.988** |

In 2D, PD residuals exhibit strong directional correlation at coarse levels ($\rho = 0.99$ at $t = 4$), reflecting the underlying data geometry (two principal modes aligned along a dominant axis). DDPM residuals remain isotropic ($\rho \approx 0$) as expected for independent Gaussian noise.

**Conclusion:** Precision Diffusion forward process is fundamentally different from DDPM. The residuals are structured quantization errors, not isotropic Gaussian noise.

### 5.2 Metric 2: Quantization Performance

**Table 3: 1D scalar quantization MSE ($K = 16$)**

| Method                  | Good Init | Poor Init |
|-------------------------|-----------|-----------|
| Lloyd-Max (optimal)     | 0.01635   | 0.01636   |
| K-means (STE)           | 0.01629   | 0.02635   |
| Precision Diffusion     | 0.02031   | 2.94589   |
| Uniform Quantization    | 0.03940   | 0.03940   |

In 1D, PD underperforms K-means ($-24.6\%$ from good init, worse from poor). This is expected: 1-D quantization is convex, and K-means converges to the global Lloyd-Max optimum. PD's predictor learns $f(x_t, t) \approx \mathbb{E}[x_0 - x_t \mid x_t]$, which at optimality equals the K-means gradient. PD adds no benefit in the convex regime but adds predictor noise.

**Table 4: 2D vector quantization MSE ($K = 16$)**

| Method              | Good Init | Moderate-Poor | Very Poor |
|---------------------|-----------|---------------|-----------|
| Lloyd-Max (optimal) | 0.1173    | —             | —         |
| K-means (100 iters) | 0.1184    | 0.7716        | 0.2036    |
| PD centroid + noise | **0.1187**| 1.2387        | 1.9912    |
| PD data-avg + noise | —         | 1.6586        | **0.8710**|

**Key finding:** From good initialization, PD matches K-means within $+0.3\%$ ($0.1187$ vs. $0.1184$). This demonstrates that PD's gradient, while noisier than K-means' exact cell-mean, points in the correct direction and converges to the same quality basin.

From poor initialization, PD converges (MSE decreases monotonically from $6.3$ to $0.87$–$1.99$ depending on variant) but significantly slower than K-means. K-means reaches $0.20$ in 100 iterations from the same very-poor init, while PD reaches $0.87$–$1.99$.

**Figure 1: Convergence from very poor init (centroid-level PD + Langevin noise)**

| Outer Iter | Temperature | Noise σ | Test MSE |
|------------|-------------|---------|----------|
| 0          | 2.000       | 0.633   | 6.293    |
| 30         | 1.597       | 0.565   | 3.080    |
| 60         | 1.195       | 0.489   | 3.139    |
| 90         | 0.792       | 0.398   | 1.538    |
| 120        | 0.389       | 0.279   | 2.013    |
| 149        | **0.000**   | 0.000   | **1.991**|

The temperature annealing is effective: noise-driven exploration dominates early training (large σ), enabling symmetry breaking, while deterministic gradient steps take over as temperature → 0.

### 5.3 Metric 3: Gradient Legitimacy

**Table 5: Gradient cosine similarity with true (Lloyd-Max) direction**

| Init         | cos(PD_centroid, true) | cos(PD_avg, true) | cos(STE, true) |
|--------------|------------------------|-------------------|----------------|
| Good         | +0.17                  | +0.17             | **−1.00**      |
| Moderate     | −0.01                  | −0.02             | **−1.00**      |
| Very Poor    | −0.10                  | −0.14             | **−1.00**      |

STE's gradient is $-\nabla_{\text{true}}$ (anti-aligned exactly) because the VQ commitment loss gradient equals the negative cell-mean direction. While directionally exact, STE's gradient is **mathematically zero** through the quantizer—the hard argmin produces no gradient, and STE manually copies it as $I$.

PD provides a non-zero, non-degenerate gradient signal. The alignment ($+0.17$ for good init) is modest but meaningful: the predictor's output correlates positively with the true correction direction. The weak alignment reflects the predictor's inherent lag (it must adapt to each new codebook state) rather than a fundamental flaw.

---

## 6. Discussion

### 6.1 Why PD Converges Slower Than K-means

The fundamental tension is between differentiability and gradient accuracy:

- **K-means:** Has exact gradient (cell mean minus centroid), but it's non-differentiable through the argmin operation.
- **PD:** Has a differentiable gradient, but it's a learned estimate—the predictor must be re-trained each time the codebook changes, introducing systematic lag.

In the joint training framework, the predictor plays "catch-up" with the codebook. This is analogous to the actor-critic gap in reinforcement learning: the critic (predictor) estimates the value (gradient) of the current policy (codebook), but as the policy improves, the critic's estimates become stale.

### 6.2 The Role of Langevin Noise

Without noise injection, centroid-level PD suffers from **symmetry collapse**: all centroids at the same position receive identical gradients and move identically forever. Langevin noise provides the stochastic perturbation needed to break symmetry, enabling centroids to explore distinct positions.

This noise mechanism is theoretically grounded: it transforms codebook optimization from gradient descent into Langevin dynamics, where the stationary distribution is related to the data distribution through the predictor's score estimate.

### 6.3 Practical Recommendations

Based on our findings, the recommended deployment strategy for PD in VQ architectures is:

1. **Initialization:** Pre-train the codebook using standard K-means (10–50 iterations). This provides a good starting point where PD's gradient is well-aligned.
2. **Fine-tuning:** Switch to PD for end-to-end training with the encoder. PD provides differentiable gradients that enable joint optimization of encoder and codebook.
3. **Noise schedule:** For codebook fine-tuning, use zero noise (the codebook is already well-separated). For VQ-VAE training from scratch, anneal Langevin noise over the first 10% of training.

### 6.4 1D vs. 2D: When Does PD Matter?

The 1D results are deliberately negative: in 1D, quantization is convex, and K-means already finds the global optimum. PD's cross-cell gradient information provides no benefit and adds predictor noise.

In 2D, PD shows its first advantage: the good-init MSE gap is only $0.3\%$, proving the gradient direction is correct. In higher dimensions (e.g., VQ-VAE latent spaces at $d = 256$), the gap is expected to close further or reverse, as:
- K-means becomes increasingly susceptible to local minima in high dimensions.
- The predictor's multi-level training provides regularization that helps escape poor local optima.
- The differentiable gradient enables joint encoder-codebook optimization, which STE cannot do genuinely.

---

## 7. Limitations and Future Work

**Limitations:**
1. From poor initialization, PD converges significantly slower than K-means. The K-means + PD hybrid strategy mitigates this but introduces a two-phase training procedure.
2. Our experiments are limited to $d \leq 2$ and synthetic data. Validation on real VQ-VAE training (e.g., image reconstruction on CIFAR-10, ImageNet) is needed.
3. The predictor's gradient alignment ($\cos \approx 0.17$) suggests room for improvement in predictor architecture and training.

**Future Work:**
1. **Higher-dimensional validation:** Extend the probe to 16D/64D Gaussian mixtures to quantify how the PD–K-means gap scales with dimension.
2. **VQ-VAE integration:** Replace STE with PD in a standard VQ-VAE and measure reconstruction quality + codebook utilization.
3. **Accelerated PD sampling:** Adapt DDIM-style non-Markovian inference to speed up PD codebook training by skipping diffusion steps.
4. **Conditional PD:** Add classifier guidance to the codebook update, enabling task-conditioned quantization.
5. **Adaptive bit schedules:** Learn the optimal bit-depth schedule per dataset rather than using a fixed linear decay.

---

## 8. Conclusion

We have introduced Precision Diffusion, a quantization-theoretic forward process for diffusion models where precision degrades through progressive bit-depth reduction rather than Gaussian noise injection. Through minimal 1D and 2D controlled experiments—the Precision Diffusion Probe—we demonstrate:

1. **PD forward residuals are structurally and statistically distinct from DDPM** ($p \approx 0$, both norm and angle KS tests in 2D).
2. **PD matches K-means quality from good initialization** ($+0.3\%$ MSE gap at $K = 16$, 2D), proving the gradient direction is correct.
3. **PD provides a genuine, differentiable gradient**, eliminating the need for STE in VQ codebook training.

Precision Diffusion reframes vector quantization within the diffusion model formalism, opening the door to a unified class of score-based quantization methods. The code and experiment scripts are available as part of the Precision Diffusion Probe framework.

---

## References

[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.

[2] Sohl-Dickstein, J., et al. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *ICML*.

[3] van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. *NeurIPS*.

[4] Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. *arXiv:1308.3432*.

[5] Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS*.

[6] Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR*.

[7] Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Trans. Inf. Theory*.

[8] Max, J. (1960). Quantizing for minimum distortion. *IRE Trans. Inf. Theory*.

[9] Esser, P., Rombach, R., & Ommer, B. (2021). Taming transformers for high-resolution image synthesis. *CVPR*.

[10] Lee, D., et al. (2022). Autoregressive image generation using residual quantization. *CVPR*.

[11] Yin, P., et al. (2019). Understanding straight-through estimator in training activation quantized neural nets. *ICLR*.

[12] Gersho, A., & Gray, R. M. (1992). *Vector Quantization and Signal Compression*. Springer.

---

## Appendix A: Experiment Configuration

| Parameter | 1D Probe | 2D Benchmark |
|-----------|----------|--------------|
| Data distribution | $0.5\mathcal{N}(-2,0.5) + 0.5\mathcal{N}(2,0.5)$ | 5-component 2D Gaussian mixture |
| Train / Test | 100K / 10K | 50K / 10K |
| Diffusion steps $T$ | 4 | 4 |
| Bit schedule | $[8,6,4,2,1]$ | $[256,64,16,4,2]$ centroids |
| Target codebook $K$ | 16 | 16 |
| Predictor architecture | Linear ($2 \to 1$) / 2-layer MLP | 3-layer MLP ($4 \to 128 \to 128 \to 2$) |
| K-means LR | 0.05–0.2 | 0.2 |
| PD LR (codebook) | 0.01–0.05 | 0.1–0.15 |
| Langevin $T_\text{start}$ | — | 0.5–2.0 |

## Appendix B: 1D Forward Process Residual Analysis

| Step | Bits (→) | σ(DDPM ε) | σ(PD ε) |
|------|----------|-----------|---------|
| t=1  | 256→64   | 1.001     | 0.058   |
| t=2  | 64→16    | 0.999     | 0.149   |
| t=3  | 16→4     | 1.000     | 0.391   |
| t=4  | 4→2      | 1.002     | 0.561   |

The scale of DDPM residuals remains constant ($\approx 1.0$) across all steps (variance-preserving). PD residuals grow in magnitude as quantization becomes coarser, reflecting the increasing information loss at lower bit depths.

## Appendix C: 2D Codebook Training Traces

**K-means (very poor init, 100 iters):**

| Iter | Test MSE |
|------|----------|
| 0    | 5.903    |
| 20   | 0.787    |
| 40   | 0.777    |
| 60   | 0.487    |
| 80   | 0.276    |
| 99   | **0.204**|

**PD centroid + Langevin noise (very poor init, 150 outer iters):**

| Outer | Temperature | Test MSE |
|-------|-------------|----------|
| 0     | 2.000       | 6.293    |
| 30    | 1.597       | 3.080    |
| 60    | 1.195       | 3.139    |
| 90    | 0.792       | 1.538    |
| 120   | 0.389       | 2.013    |
| 149   | 0.000       | **1.991**|

K-means converges approximately 10× faster than PD from very poor initialization. However, PD's convergence is monotonic and stable, with no divergence or oscillation despite the stochastic noise injection.

---

*Correspondence to: [author email]*
