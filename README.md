# PINN4SOH

Physics-Informed Neural Networks for Battery State-of-Health Prediction

This code accompanies the paper: [Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis](https://www.nature.com/articles/s41467-024-48779-z) (Nature Communications, 2024)

## Overview

PINN4SOH implements a two-stage training approach:
1. **Pretraining**: Learn general battery degradation patterns from large datasets
2. **Fine-tuning**: Transfer knowledge to new battery domains with minimal target data

The physics-informed approach incorporates PDE residual constraints and monotonicity constraints for physically realistic predictions.

## Project Structure

```
PINN4SOH/
├── src/                    # Source code
│   ├── models.py           # PINN architecture
│   ├── data_loaders.py     # Dataset loaders
│   └── train_pretrain.py   # Pretraining script
├── data/                   # Preprocessed datasets
├── pretrained model/       # Pretrained model weights
├── dataloader/             # Data loading utilities
├── utils/                  # Utility functions
└── requirements.txt        # Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch >= 2.0.0
- scikit-learn >= 0.24.2
- numpy >= 1.20.3
- pandas >= 1.3.5
- matplotlib >= 3.3.4

## Installation

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Pretraining

Train a PINN model on a specific dataset:

```bash
# Train on XJTU dataset (batch 0 = 2C charge rate)
python3 src/train_pretrain.py --dataset XJTU --batch 0

# Train on MIT dataset
python3 src/train_pretrain.py --dataset MIT --batch 1

# Train on TJU dataset
python3 src/train_pretrain.py --dataset TJU --batch 1
```

Options:
- `--dataset`: XJTU, MIT, HUST, or TJU
- `--batch`: Batch index (dataset-specific)
- `--epochs`: Number of training epochs (default: 200)

### Output

Training outputs are saved to `results/{dataset}_batch{batch}/`:
- `model.pth`: Trained model weights
- `true_label.npy`: Ground truth SOH values
- `pred_label.npy`: Predicted SOH values
- `logging.txt`: Training log

## Datasets

Preprocessed data for 387 batteries is included in `./data/`:
- XJTU: 55 NCM batteries
- MIT: 125 LFP batteries
- HUST: 77 batteries
- TJU: 130 NCA/NCM batteries

Raw data sources:
- XJTU: https://wang-fujin.github.io/
- TJU: https://zenodo.org/record/6405084
- HUST: https://data.mendeley.com/datasets/nsc7hnsg4s/2
- MIT: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

Preprocessing code: https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library

## Citation

```bibtex
@article{wang2024physics,
  title={Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis},
  author={Wang, Fujin and Zhai, Zhi and Zhao, Zhibin and Di, Yi and Chen, Xuefeng},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={4332},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
