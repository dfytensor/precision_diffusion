# Precision Diffusion

> 精度扩散：一种替代高斯扩散的量化理论框架，用于可微向量量化训练  
> High-Dimensional Validation (d=2048) + Real Image Compression Application

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-red.svg)](https://pytorch.org/)

---

## 概述

本仓库包含两大部分：

1. **原始精度扩散论文与探针实验** (d=1, 2 合成数据)
2. **高维验证与真实图像压缩应用** (d=2048 真实图像隐变量)

精度扩散 (Precision Diffusion, PD) 提出以**逐级降低量化位宽**替代高斯加噪作为扩散模型的前向过程。产生的残差是结构化的**量化误差**，而非各向同性高斯噪声。这使其能够为 VQ-VAE 等架构的向量量化训练提供完全可微的梯度路径，替代传统的直通估计器 (STE)。

## 核心发现

| 发现 | 详情 |
|------|------|
| PD 前向过程 ≠ DDPM | d=2048 上 KS 检验 p=0，量化残差具有结构性相关 |
| PD 量化质量 ≈ K-means | K-means 预训练 + PD 微调，差距仅 +0.07% |
| PD 梯度在高维有效 | cos(PD, true) = +0.234 (K=64)，超过论文 d=2 的 +0.17 |
| Langevin 噪声维度灾难 | 原始公式在高维失效，修复方案：1/√d 缩放 |
| RVQ 真实图像压缩 | 相同 BPP 下 +6.76 dB，或 −20% BPP @ −0.43 dB |

## 率-质量对比

```
PSNR (dB)
  40 ┤                                    ● Uniform 8b (39.45)
     │                          ● RVQ 4级 (39.02)
  39 ┤
     │                    ● RVQ 3级 (38.03)    ● Uniform 6b (37.77)
  38 ┤
     │          ● RVQ 2级 (36.37)
  36 ┤
  32 ┤  ● Uniform 4b (32.26)
     └────┬─────┬─────┬─────┬─────┬─────┬──── BPP
         28    30    32    34    36    38    40
```

## 快速开始

### 环境要求

- Python 3.12
- PyTorch 2.6.0+ (CUDA 12.4)
- NumPy, SciPy, scikit-learn, Pillow
- NVIDIA GPU (≥ 12 GB VRAM)

### 克隆

```bash
git clone https://github.com/dfytensor/precision_diffusion.git
cd precision_diffusion
```

### 运行原始探针 (d=1, 2)

```bash
# 1D 标量量化探针
python precision_diffusion_probe.py

# 2D 向量量化探针
python precision_diffusion_probe_2d.py

# 2D 联合训练 + Langevin 噪声
python precision_diffusion_probe_2d_joint.py
```

### 运行高维验证 (d=2048)

```bash
# 全 K 值扫描验证 (断言 1-3)
python pd_validate_on_v10_v4.py

# Langevin 噪声修复实验
python pd_validate_on_v10_v3.py

# 高质量 RVQ 压缩对比 (含解码器微调)
python pd_hq_comparison.py
```

> **注意**: 高维验证脚本需要 v10 编解码器的预训练模型。请将以下文件放置到对应位置：
> - `F:\OpenASH\vision_voc\uniencode\periodic_codec_checkpoints\codec_residual.pt`
> - `F:\OpenASH\vision_voc\uniencode\v10_bundle\spatial_residual_models_v10.pt`
> - `F:\OpenASH\vision_voc\uniencode\v10_bundle\spatial_residual_models_v9_resblock.pt`
> - `F:\OpenASH\vision_voc\uniencode\causal_models_v7.pt`
> - `F:\OpenASH\vision_voc\uniencode\causal_models_imgpred.pt`

## 仓库结构

```
precision_diffusion/
│
├── 📄 论文文档
│   ├── precision_diffusion_paper.md            # 原论文 (英文)
│   ├── precision_diffusion_paper_cn.md         # 原论文 (中文)
│   ├── PD_HIGH_DIM_VALIDATION.md               # 高维验证论文 ★
│   └── PD_V10_VALIDATION_REPORT.md             # 验证报告 (v1-v3)
│
├── 🔬 原始探针 (d=1, 2)
│   ├── precision_diffusion_probe.py            # 1D 标量量化
│   ├── precision_diffusion_probe_2d.py         # 2D 向量量化
│   └── precision_diffusion_probe_2d_joint.py   # 2D 联合训练
│
├── 🧪 高维验证脚本 (d=2048)
│   ├── pd_validate_on_v10.py                   # v1: PCA 预测器
│   ├── pd_validate_on_v10_v2.py                # v2: 全维深度预测器
│   ├── pd_validate_on_v10_v3.py                # v3: Langevin 噪声修复
│   └── pd_validate_on_v10_v4.py                # v4: 全 K 值扫描
│
├── 🖼️ 真实图像压缩应用
│   ├── pd_real_application_v10.py              # 位置级 VQ 应用
│   ├── pd_rvq_application.py                   # RVQ 应用 (无微调)
│   └── pd_hq_comparison.py                     # 高质量 RVQ + 解码器微调 ★
│
├── 📊 实验结果
│   ├── v10_validation_output/                  # 验证数据 (JSON, NPZ)
│   ├── v10_hq_comparison/                      # 高质量对比图像
│   └── v10_rvq_application/                    # RVQ 应用结果
│
└── .gitignore
```

## 实验结果详解

### 断言一：前向过程统计差异

| 步骤 | PD 残差范数 | DDPM 范数 | PD 维间 \|corr\| | DDPM 维间 \|corr\| | p 值 |
|------|-----------|----------|----------------|------------------|------|
| t=1 | 0.11 | 45.3 | 0.024 | 0.018 | 0 |
| t=2 | 0.43 | 45.3 | 0.026 | 0.018 | 0 |
| t=3 | 1.81 | 45.2 | **0.064** | 0.016 | 0 |
| t=4 | 9.96 | 45.3 | **0.099** | 0.018 | 0 |

### 断言二：量化器训练质量

| 方法 | K=16 | K=64 | K=256 |
|------|------|------|-------|
| K-means | 16.321 | 16.084 | 16.001 |
| **PD 微调** | **+0.02%** | **+0.07%** | **+0.11%** |
| PD 随机初始化 | +1.74% | +2.33% | +2.72% |

### Langevin 噪声修复

| 策略 | 噪声/梯度比 | MSE | vs K-means |
|------|-----------|-----|-----------|
| 原始公式 | 16× | 306.30 | +1786% |
| **1/√d 缩放** | **0.3×** | **16.64** | **+2.4%** |
| **自适应缩放** | **0.5×** | **16.36** | **+0.7%** |

### RVQ 图像压缩

| 方案 | PSNR | BPP | vs Uniform 8b |
|------|------|-----|--------------|
| Uniform 8b (基线) | 39.45 dB | 40.09 | — |
| **RVQ 4 级 + 微调** | **39.02 dB** | **32.09** | **−0.43 dB, −20% BPP** |
| Uniform 4b | 32.26 dB | 32.09 | −7.19 dB |

## 引用

**BibTeX:**

```bibtex
@article{pd_high_dim_validation_2026,
  title     = {High-Dimensional Validation of Precision Diffusion and Its Application to Real Image Compression},
  author    = {dfytensor},
  year      = {2026},
  month     = {July},
  url       = {https://github.com/dfytensor/precision_diffusion}
}
```

**中文:**

> dfytensor. (2026). 精度扩散的高维验证与真实图像压缩应用. GitHub 仓库. https://github.com/dfytensor/precision_diffusion

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。
