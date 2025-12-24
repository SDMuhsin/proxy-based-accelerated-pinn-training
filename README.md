# Frozen Physics Proxies for Accelerating Cross-Chemistry Battery State-of-Health Transfer Learning

Code for the paper under review at Applied Intelligence (Springer).

## Overview

Battery state-of-health prediction across different chemistries requires efficient transfer learning methods. Physics-informed neural networks improve accuracy by encoding degradation dynamics as differential equation constraints, but require expensive automatic differentiation during adaptation.

We investigate whether frozen physics networks trained on source domains can serve as effective regularizers during transfer learning when evaluated using finite difference approximations rather than exact derivatives.

## Key Findings

Through 24 experiments across three lithium-ion chemistries (NCM, NCA, LFP):

1. Finite difference approximations achieve 1.3-2.1x training speedup
2. Simple forward differences match higher-order Richardson extrapolation because stochastic gradient noise dominates approximation error
3. Target dataset size is the primary predictor of effectiveness (r=-0.891, p=0.017)
4. Physics constraints provide 5-10x accuracy improvements over data-driven methods even when approximately evaluated

## Requirements

- Python 3.10+
- PyTorch >= 2.0.0
- scikit-learn >= 0.24.2
- numpy >= 1.20.3
- pandas >= 1.3.5
- matplotlib >= 3.3.4

## Installation

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### Pretraining

```bash
python3 src/train_pretrain.py --dataset XJTU --batch 0
python3 src/train_pretrain.py --dataset MIT --batch 1
python3 src/train_pretrain.py --dataset TJU --batch 1
```

### Transfer Learning with Physics Proxy

```bash
python3 src/physics_approximation_strategies.py
```

## Datasets

This work uses four public battery datasets:

- XJTU (55 NCM batteries): https://wang-fujin.github.io/
- MIT (125 LFP batteries): https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
- TJU (130 NCA/NCM batteries): https://zenodo.org/record/6405084
- HUST (77 batteries): https://data.mendeley.com/datasets/nsc7hnsg4s/2

## Acknowledgment

This code builds on the PINN4SOH framework from Wang et al. (Nature Communications, 2024).

## Citation

Please contact ckp908@usask.ca for citation. This work is currently under review.
