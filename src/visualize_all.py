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
import torch

from src.visualize import (
    plot_training_curves,
    plot_width_histogram,
    plot_roc_pr_curves,
    multi_model_grid,
)
from src.evaluate import load_config, load_checkpoint, evaluate_model, find_checkpoint
from src.datasets import load_split, get_dataloader


def find_file(relative_path):
    """Search for a file locally or under mounted Google Drive project paths."""
    candidates = [
        relative_path,
        os.path.join('/content/drive/MyDrive/coronary-vessel-segmentation', relative_path),
        os.path.join('/content/drive/MyDrive', relative_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def generate_qualitative_comparisons():
    """Evaluate all available trained checkpoints on the test set and plot multi-model visual comparisons."""
    models_info = [
        ('configs/baseline.yaml',    'Baseline U-Net (M1)'),
        ('configs/hybrid.yaml',      'Hybrid U-Net (M2)'),
        ('configs/hybrid_i2.yaml',   'Hybrid Mamba+I2 (Proposed)'),
    ]

    available_models = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for cfg_path, label in models_info:
        if os.path.exists(cfg_path):
            cfg = load_config(cfg_path)
            ckpt_path = find_checkpoint(cfg)
            if ckpt_path is not None:
                available_models.append((cfg, label))

    if not available_models:
        print("[Viz] No checkpoints found yet under results/checkpoints/ or Google Drive. Skipping image comparison grids.")
        return

    cfg0 = available_models[0][0]
    data_root = cfg0['data_root']
    split_file = cfg0['split_file']

    if not os.path.exists(data_root):
        drive_candidates = [
            '/content/drive/MyDrive/coronary-vessel-segmentation/data/DCA1',
            '/content/drive/MyDrive/DCA1_data',
            '/content/drive/MyDrive/Database_134_Angiograms',
            '/content/drive/MyDrive/DCA1',
        ]
        for dc in drive_candidates:
            if os.path.exists(dc):
                data_root = dc
                break

    if not os.path.exists(split_file) or not os.path.exists(data_root):
        print(f"[Viz] Dataset or split file missing at {data_root}. Skipping qualitative grids.")
        return

    loader = get_dataloader(data_root, split_file, split='test', batch_size=1, num_workers=2)

    model_preds_dict = {}
    model_probs_dict = {}
    model_metrics_dict = {}
    raw_images = []
    raw_gts = []
    image_ids = []

    for cfg, label in available_models:
        print(f"[Viz] Generating predictions for {label}...")
        try:
            model = load_checkpoint(cfg, device)
            per_img, probs, preds, gts, ids = evaluate_model(model, loader, device, cfg)
            model_preds_dict[label] = preds
            model_probs_dict[label] = probs
            model_metrics_dict[label] = per_img

            if not raw_images:
                for img_id in ids:
                    p = os.path.join(data_root, 'images', f'{img_id}.pgm')
                    if os.path.exists(p):
                        img_np = np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0
                    else:
                        img_np = np.zeros((300, 300), dtype=np.float32)
                    raw_images.append(img_np)
                raw_gts = gts
                image_ids = ids
        except Exception as e:
            print(f"[Viz] Could not evaluate {label}: {e}")

    if not model_preds_dict or not raw_images:
        return

    # 1. Multi-model grid for best/representative test samples
    multi_model_grid(
        images=raw_images,
        gts=raw_gts,
        model_preds_dict=model_preds_dict,
        ids=image_ids,
        out_path='results/figures/multi_model_comparison.png',
        n_show=min(6, len(raw_images)),
        model_metrics_dict=model_metrics_dict
    )

    # 2. Individual test sample comparison images
    out_cases_dir = 'results/figures/qualitative'
    os.makedirs(out_cases_dir, exist_ok=True)
    for i in range(min(10, len(raw_images))):
        case_id = image_ids[i]
        single_preds = {lbl: [model_preds_dict[lbl][i]] for lbl in model_preds_dict}
        single_metrics = {lbl: [model_metrics_dict[lbl][i]] for lbl in model_metrics_dict}
        multi_model_grid(
            images=[raw_images[i]],
            gts=[raw_gts[i]],
            model_preds_dict=single_preds,
            ids=[case_id],
            out_path=os.path.join(out_cases_dir, f'case_{case_id}.png'),
            n_show=1,
            model_metrics_dict=single_metrics
        )
    print(f"[Viz] Saved individual qualitative sample comparisons in {out_cases_dir}/")

    # 3. Combined ROC and PR curves
    if len(model_probs_dict) >= 1:
        prob_list = [model_probs_dict[lbl] for lbl in model_probs_dict]
        lbl_list = list(model_probs_dict.keys())
        plot_roc_pr_curves(prob_list, raw_gts, lbl_list, out_path='results/figures/roc_pr.png')


def main():
    os.makedirs('results/figures', exist_ok=True)

    # 1. Training curves
    experiments = [
        ('results/runs/baseline_unet/metrics.csv', 'Baseline U-Net (M1)'),
        ('results/runs/hybrid_unet/metrics.csv',   'Hybrid U-Net (M2)'),
        ('results/runs/hybrid_unet_i2/metrics.csv','Hybrid U-Net + I2 (Proposed)'),
    ]
    csv_paths = []
    labels    = []
    for rel_path, lbl in experiments:
        found = find_file(rel_path)
        if found:
            csv_paths.append(found)
            labels.append(lbl)

    if csv_paths:
        plot_training_curves(csv_paths, labels, out_path='results/figures/training_curves.png')
    else:
        print("[Viz] No metrics.csv found yet under results/runs/ or Google Drive.")

    # 2. Vessel width histogram from training GT
    split_file = 'splits/split.json'
    mask_dir = find_file('data/DCA1/masks') or 'data/DCA1/masks'
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

    # 3. Qualitative multi-model comparison grids on test set
    print("[Viz] Generating multi-model prediction comparisons...")
    generate_qualitative_comparisons()

    print("[Viz] Visualizations check complete.")


if __name__ == '__main__':
    main()

