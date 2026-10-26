"""
Precompute skeleton (centerline) GT masks for all images.
Run once before training M2+I2 (use_innovation_i2=True).

Usage: python -m src.precompute_skeletons

Output: data/DCA1/skeletons/{id}_skel.pgm  for each image
"""
import os, json
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


def main():
    data_root  = 'data/DCA1'
    split_file = 'splits/split.json'
    mask_dir   = os.path.join(data_root, 'masks')
    skel_dir   = os.path.join(data_root, 'skeletons')
    os.makedirs(skel_dir, exist_ok=True)

    with open(split_file) as f:
        split = json.load(f)

    all_ids = split['train'] + split['val'] + split['test']
    print(f"Precomputing skeletons for {len(all_ids)} images...")

    for img_id in all_ids:
        mask_path = os.path.join(mask_dir, f'{img_id}_gt.pgm')
        skel_path = os.path.join(skel_dir, f'{img_id}_skel.pgm')

        if os.path.exists(skel_path):
            continue  # skip if already computed

        mask = np.array(Image.open(mask_path).convert('L'))
        binary = (mask > 0)
        skel = skeletonize(binary).astype(np.uint8) * 255

        Image.fromarray(skel).save(skel_path)

    n = len(os.listdir(skel_dir))
    print(f"Done. Skeletons saved to {skel_dir}/ ({n} files)")


if __name__ == '__main__':
    main()

