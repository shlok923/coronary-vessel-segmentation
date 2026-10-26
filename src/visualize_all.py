"""
Batch visualization script: generates training curves, ROC/PR curves,
qualitative comparison grids, and vessel width histograms from experiment results.

Usage:
    python -m src.visualize_all
"""
import os
import glob
import pandas as pd
from PIL import Image
import numpy as np

from src.visualize import (
    plot_training_curves,
    plot_width_histogram,
)


def main():
    os.makedirs('results/figures', exist_ok=True)

    # 1. Training curves
    experiments = [
        ('results/runs/baseline_unet/metrics.csv', 'Baseline U-Net (M1)'),
        ('results/runs/hybrid_unet/metrics.csv',   'Hybrid U-Net (M2)'),
        ('results/runs/hybrid_unet_i2/metrics.csv','Hybrid U-Net + I2 (Proposed)'),
    ]
    csv_paths = [p for p, _ in experiments if os.path.exists(p)]
    labels    = [lbl for p, lbl in experiments if os.path.exists(p)]

    if csv_paths:
        plot_training_curves(csv_paths, labels, out_path='results/figures/training_curves.png')
    else:
        print("[Viz] No metrics.csv found yet under results/runs/.")

    # 2. Vessel width histogram from training GT
    split_file = 'splits/split.json'
    mask_dir = 'data/DCA1/masks'
    if os.path.exists(split_file) and os.path.exists(mask_dir):
        import json
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize

        with open(split_file) as f:
            split = json.load(f)

        widths_list = []
        for img_id in split.get('train', []):
            msk_path = os.path.join(mask_dir, f'{img_id}_gt.pgm')
            if os.path.exists(msk_path):
                mask = np.array(Image.open(msk_path).convert('L')) > 0
                skel = skeletonize(mask)
                if skel.sum() > 0:
                    dt = distance_transform_edt(mask)
                    widths = 2.0 * dt[skel]
                    widths_list.extend(widths.tolist())

        if widths_list:
            plot_width_histogram([widths_list], ['DCA1 Train GT'], thin_threshold=3,
                                 out_path='results/figures/width_histogram.png')

    print("[Viz] Visualizations check complete.")


if __name__ == '__main__':
    main()
