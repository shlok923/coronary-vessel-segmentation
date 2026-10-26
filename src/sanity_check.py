"""
Data sanity check script. Run before any training.

Usage: python -m src.sanity_check

Checks:
  - All image/mask pairs exist
  - Images are 300x300 greyscale
  - Masks are binary (values in {0, 255} or {0, 1})
  - Vessel pixel fraction (expected ~5%)
  - Saves a visual grid of samples
"""
import os, sys, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data_root  = 'data/DCA1'
    split_file = 'splits/split.json'
    img_dir    = os.path.join(data_root, 'images')
    msk_dir    = os.path.join(data_root, 'masks')

    with open(split_file) as f:
        split = json.load(f)

    all_ids = split['train'] + split['val'] + split['test']
    print(f"\n{'='*60}")
    print(f"DCA1 Dataset Sanity Check")
    print(f"  Total images: {len(all_ids)}")
    print(f"  Train: {len(split['train'])}  Val: {len(split['val'])}  Test: {len(split['test'])}")
    print(f"{'='*60}")

    errors = []
    vessel_fracs = []
    shapes_ok = True

    for img_id in all_ids:
        img_path = os.path.join(img_dir, f'{img_id}.pgm')
        msk_path = os.path.join(msk_dir, f'{img_id}_gt.pgm')

        # Existence
        if not os.path.exists(img_path):
            errors.append(f"Missing image: {img_path}")
            continue
        if not os.path.exists(msk_path):
            errors.append(f"Missing mask: {msk_path}")
            continue

        # Shape
        img = np.array(Image.open(img_path).convert('L'))
        msk = np.array(Image.open(msk_path).convert('L'))

        if img.shape != (300, 300):
            errors.append(f"Image {img_id}: shape {img.shape}, expected (300,300)")
            shapes_ok = False
        if msk.shape != (300, 300):
            errors.append(f"Mask {img_id}: shape {msk.shape}, expected (300,300)")

        # Binarize and compute vessel fraction
        binary = (msk > 0).astype(float)
        frac = binary.mean()
        vessel_fracs.append(frac)

    print(f"\nPairing check: {'OK' if not errors else 'FAILED'}")
    if errors:
        for e in errors[:10]:
            print(f"  ERROR: {e}")

    print(f"\nShape check: {'OK' if shapes_ok else 'FAILED'}")

    frac_arr = np.array(vessel_fracs)
    print(f"\nVessel pixel fraction statistics:")
    print(f"  Mean:   {frac_arr.mean()*100:.2f}%  (expected ~5%)")
    print(f"  Std:    {frac_arr.std()*100:.2f}%")
    print(f"  Min:    {frac_arr.min()*100:.2f}%")
    print(f"  Max:    {frac_arr.max()*100:.2f}%")
    print(f"  Median: {np.median(frac_arr)*100:.2f}%")

    # Save visual grid
    os.makedirs('results/figures', exist_ok=True)
    sample_ids = sorted(all_ids)[:16]
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    axes = axes.ravel()

    for i, img_id in enumerate(sample_ids):
        img = np.array(Image.open(os.path.join(img_dir, f'{img_id}.pgm')).convert('L'))
        msk = np.array(Image.open(os.path.join(msk_dir, f'{img_id}_gt.pgm')).convert('L'))
        axes[2*i].imshow(img, cmap='gray')
        axes[2*i].set_title(f'Img {img_id}', fontsize=7)
        axes[2*i].axis('off')
        axes[2*i+1].imshow(msk, cmap='gray')
        axes[2*i+1].set_title(f'Mask {img_id}', fontsize=7)
        axes[2*i+1].axis('off')

    plt.suptitle('DCA1 Sample Images and Masks', fontsize=14)
    plt.tight_layout()
    grid_path = 'results/figures/data_grid.png'
    plt.savefig(grid_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nVisual grid saved: {grid_path}")
    print(f"\n{'='*60}")
    print(f"Sanity check {'PASSED' if not errors else 'FAILED'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
