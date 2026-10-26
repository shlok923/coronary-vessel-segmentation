# Coronary Vessel Segmentation — Hybrid CNN–Mamba U-Net

**AI512 Project** · Binary segmentation of coronary vessels on the DCA1 angiogram dataset.

## Overview

This repository contains a full research pipeline to train and evaluate two segmentation models:

| Model | Description |
|-------|-------------|
| **M1** | Baseline 2D U-Net |
| **M2** | Hybrid CNN–Mamba U-Net |
| **M2+I2** | M2 + centerline auxiliary head + clDice loss (proposed) |

---

## Setup

### 1. Environment (local)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Optional: Mamba SSM (CUDA GPU required)
```bash
pip install causal-conv1d==1.2.0
pip install mamba-ssm==1.2.0
```
If installation fails, the pure-PyTorch SS2D fallback is used automatically.

### 3. Environment (Colab)
Open `colab_runner.ipynb` in Colab and run cells top to bottom.  
The VS Code Colab extension is recommended: install it, open this notebook, connect to a Colab runtime.

### CUDA / PyTorch versions
- PyTorch ≥ 2.1.0  
- CUDA ≥ 11.8 (for mamba-ssm)  
- Python 3.10+

---

## Data Setup

Place DCA1 data as:
```
data/DCA1/
  images/   1.pgm, 2.pgm, ..., 134.pgm
  masks/    1_gt.pgm, 2_gt.pgm, ..., 134_gt.pgm
```

The dataset split (80 train / 24 val / 30 test, seed=42) is at `splits/split.json`.

**Note:** The SRS references 130 images; this project uses all 134 available images. This deviation is documented in `splits/split.json`.

---

## Reproduce Every Result

### Data sanity check
```bash
python -m src.sanity_check
```

### Unit tests (run before any training)
```bash
pytest tests/ -v
```

### Train models
```bash
python -m src.train --config configs/baseline.yaml     # M1
python -m src.train --config configs/hybrid.yaml       # M2
python -m src.train --config configs/hybrid_i2.yaml    # M2+I2 (proposed)
```

### Evaluate on test set (run ONCE, after training)
```bash
python -m src.evaluate --all --split test
```

### Generate all figures
```bash
python -m src.visualize_all
```

### Profile efficiency
```bash
python -m src.profile --config configs/baseline.yaml
python -m src.profile --config configs/hybrid_i2.yaml
```

All outputs go to `results/`.

---

## Repository Structure

```
coronary-vessel-segmentation/
  configs/           YAML config per experiment
  data/DCA1/         images/, masks/ (user-provided)
  splits/split.json  Fixed 80/24/30 train/val/test split
  src/
    datasets.py      Dataset, augmentation, DataLoader
    losses.py        BCE, Dice, soft-skeleton, clDice losses
    metrics.py       All evaluation metrics (unit-tested)
    train.py         Training loop
    evaluate.py      Test-set evaluation and table generation
    postprocess.py   Component removal, morphological closing
    visualize.py     Figures: curves, qualitative, ROC, histogram
    profile.py       Params, FLOPs, inference time, GPU memory
    sanity_check.py  Dataset statistics and visual grid
    models/
      unet.py        Baseline U-Net (M1)
      vss_block.py   SS2D block, auto-select backend
      hybrid_unet.py Hybrid CNN-Mamba U-Net (M2), innovation flag
  tests/
    test_metrics.py  Metric unit tests (perfect, empty, broken line...)
    test_shapes.py   Model I/O shape tests
  results/           Generated tables, figures, checkpoints
  colab_runner.ipynb Colab training notebook
  requirements.txt
```

---

## Innovation I2: Centerline-Aware Training

The proposed model (M2+I2) adds:
1. **Auxiliary centerline head** — predicts vessel skeleton from decoder features
2. **clDice loss** — topology-aware loss rewarding connected vessel predictions
3. **Skeleton BCE** — supervises the auxiliary head with GT skeletons extracted dynamically from the masks

Toggle with `use_innovation_i2: true/false` in any config YAML.

---

## Attribution

The VSS/SS2D block is adapted from:
- **VM-UNet** (Ruan & Xiang, 2024) · https://github.com/JCruan519/VM-UNet
- **VMamba** (Liu et al., 2024) · https://github.com/MzeroMiko/VMamba  
Used under Apache 2.0 license. Code restructured for this project.

---

## References

1. Cervantes-Sanchez et al. (2019). *Automatic Segmentation of Coronary Arteries in X-ray Angiograms.* — DCA1 dataset reference.
2. Gu et al. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
3. Wang et al. (2024). *Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation.*
4. Shit et al. (2021). *clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.*

