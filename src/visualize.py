"""
Visualization utilities for coronary vessel segmentation.

Generates:
  - Qualitative comparison grids (image | GT | M1 | M2 | error map)
  - Training curves from CSV logs
  - ROC and PR curves
  - Vessel width histogram
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd


def error_map(pred, gt):
    """
    Create a colour-coded error map: TP=green, FP=red, FN=blue, TN=black.
    Returns (H,W,3) uint8 RGB array.
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    H, W = pred.shape
    rgb  = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[pred & gt]   = [0,   200,   0]   # TP - green
    rgb[pred & ~gt]  = [200,   0,   0]   # FP - red
    rgb[~pred & gt]  = [0,     0, 200]   # FN - blue
    # TN stays black
    return rgb


def qualitative_grid(images, gts, preds_m1, preds_m2, ids,
                      sort_by_dice=True, n_show=6, out_path='results/figures/qualitative.png',
                      dice_scores_m1=None, dice_scores_m2=None):
    """
    Plot a grid: [Image | GT | M1 | M2 | Error(M1) | Error(M2)] for best/median/worst cases.
    """
    n = min(n_show, len(images))
    fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Image', 'Ground Truth', 'M1 Pred', 'M2+I2 Pred',
                  'M1 Error', 'M2+I2 Error']
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10, fontweight='bold')

    for i in range(n):
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 1].imshow(gts[i],    cmap='gray')
        axes[i, 2].imshow(preds_m1[i], cmap='gray')
        axes[i, 3].imshow(preds_m2[i], cmap='gray')
        axes[i, 4].imshow(error_map(preds_m1[i], gts[i]))
        axes[i, 5].imshow(error_map(preds_m2[i], gts[i]))

        label = f"ID={ids[i]}"
        if dice_scores_m1:
            label += f" | M1 Dice={dice_scores_m1[i]:.3f}"
        if dice_scores_m2:
            label += f" | M2 Dice={dice_scores_m2[i]:.3f}"
        axes[i, 0].set_ylabel(label, fontsize=8, rotation=0, ha='right', va='center')

        for j in range(6):
            axes[i, j].axis('off')

    # Legend
    patches = [
        mpatches.Patch(color=[0, 200/255, 0], label='TP'),
        mpatches.Patch(color=[200/255, 0, 0], label='FP'),
        mpatches.Patch(color=[0, 0, 200/255], label='FN'),
    ]
    fig.legend(handles=patches, loc='lower right', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Saved qualitative grid: {out_path}")


def plot_training_curves(csv_paths, labels, out_path='results/figures/training_curves.png'):
    """Plot train/val loss and val Dice curves from multiple experiment CSVs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for csv_path, label in zip(csv_paths, labels):
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        axes[0].plot(df['epoch'], df['train_loss'],  label=f'{label} train')
        axes[1].plot(df['epoch'], df['val_loss'],    label=f'{label} val')
        axes[2].plot(df['epoch'], df['val_dice'],    label=label)

    for ax, title, ylabel in zip(axes,
                                  ['Train Loss', 'Val Loss', 'Val Dice'],
                                  ['Loss', 'Loss', 'Dice']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Saved training curves: {out_path}")


def plot_roc_pr_curves(prob_maps_list, gts, labels, out_path='results/figures/roc_pr.png'):
    """Plot ROC and PR curves for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for prob_maps, label in zip(prob_maps_list, labels):
        all_probs = np.concatenate([p.ravel() for p in prob_maps])
        all_gts   = np.concatenate([g.ravel() for g in gts])

        fpr, tpr, _ = roc_curve(all_gts, all_probs)
        roc_auc     = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.3f})')

        prec, rec, _ = precision_recall_curve(all_gts, all_probs)
        pr_auc       = auc(rec, prec)
        axes[1].plot(rec, prec, label=f'{label} (AUC={pr_auc:.3f})')

    axes[0].plot([0,1],[0,1],'k--', alpha=0.5)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('PR Curve'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Saved ROC/PR curves: {out_path}")


def plot_width_histogram(width_values_list, labels, thin_threshold=3,
                          out_path='results/figures/width_histogram.png'):
    """Plot vessel width distribution for training set."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for widths, label in zip(width_values_list, labels):
        widths = np.array(widths)
        ax.hist(widths, bins=30, alpha=0.6, label=label, density=True)

    ax.axvline(x=thin_threshold, color='red', linestyle='--',
               label=f'Thin/thick boundary ({thin_threshold}px)')
    ax.set_xlabel('Vessel width (pixels)')
    ax.set_ylabel('Density')
    ax.set_title('Vessel Width Distribution (Training GT)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Saved width histogram: {out_path}")
