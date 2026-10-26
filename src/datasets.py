"""
DCA1 Coronary Angiogram Dataset loader.
- Images: greyscale 300x300 PGM, normalized to [0,1]
- Masks:  binary 300x300 PGM (>0 → 1)
- Padding: reflect-pad 300→320 (divisible by 32)
- Augmentation: flips, rotation, scale/translate, elastic, brightness/contrast/gamma
"""
import os, json, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize


def load_split(split_file):
    with open(split_file) as f:
        return json.load(f)


def get_augmentation_pipeline(training=True):
    """
    Returns an albumentations pipeline.
    Both image and mask are transformed together.
    Masks use nearest-neighbour interpolation automatically via albumentations.
    """
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=0, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0,
                               border_mode=0, p=0.5),
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ])
    else:
        return A.Compose([])  # no-op for val/test


def reflect_pad(img_np, target=320):
    """Reflect-pad a (H,W) numpy array from 300 to 320."""
    h, w = img_np.shape[:2]
    pad_h = target - h
    pad_w = target - w
    top  = pad_h // 2
    bot  = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if img_np.ndim == 2:
        return np.pad(img_np, ((top, bot), (left, right)), mode='reflect'), (top, bot, left, right)
    else:
        return np.pad(img_np, ((top, bot), (left, right), (0,0)), mode='reflect'), (top, bot, left, right)


class DCA1Dataset(Dataset):
    """
    DCA1 coronary angiogram dataset.

    Args:
        data_root (str): path to data/DCA1/ containing images/, masks/, skeletons/
        ids (list[int]): image IDs to include
        training (bool): apply augmentation if True
        use_skeleton (bool): also load precomputed skeleton masks for I2 innovation
        target_size (int): padded size (default 320)
    """
    def __init__(self, data_root, ids, training=True, use_skeleton=False, target_size=320):
        self.data_root   = data_root
        self.ids         = ids
        self.training    = training
        self.use_skeleton = use_skeleton
        self.target_size = target_size
        self.aug         = get_augmentation_pipeline(training)

    def __len__(self):
        return len(self.ids)

    def _load_pgm(self, path):
        """Load a PGM file as float32 numpy array in [0,1]."""
        img = Image.open(path).convert('L')
        return np.array(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path  = os.path.join(self.data_root, 'images', f'{image_id}.pgm')
        mask_path = os.path.join(self.data_root, 'masks',  f'{image_id}_gt.pgm')

        image = self._load_pgm(img_path)           # (300,300) float32 [0,1]
        mask  = self._load_pgm(mask_path)           # (300,300) float32 [0,1]
        mask  = (mask > 0).astype(np.float32)       # binarize

        # Augmentation (on original 300x300 before padding)
        if self.training:
            aug_input = {'image': image, 'mask': mask}
            result = self.aug(**aug_input)
            image, mask = result['image'], result['mask']

        # Reflect-pad 300 → target_size
        image_padded, pad_params = reflect_pad(image, self.target_size)
        mask_padded,  _          = reflect_pad(mask,  self.target_size)

        # Convert to tensors: (1, H, W)
        image_t = torch.from_numpy(image_padded).unsqueeze(0)  # (1,320,320)
        mask_t  = torch.from_numpy(mask_padded).unsqueeze(0)   # (1,320,320)

        item = {
            'image':      image_t,
            'mask':       mask_t,
            'image_id':   image_id,
            'pad_params': pad_params,   # (top, bot, left, right) for cropping
        }

        if self.use_skeleton:
            # Generate centerline skeleton dynamically on-the-fly from the (augmented) mask.
            # This ensures skeleton labels perfectly align with augmented vessel geometry
            # without requiring any precomputed skeleton files on disk.
            skel = skeletonize(mask > 0).astype(np.float32)
            skel_padded, _ = reflect_pad(skel, self.target_size)
            item['skeleton'] = torch.from_numpy(skel_padded).unsqueeze(0)

        return item


def unpad(tensor, pad_params, original_size=300):
    """
    Remove reflect padding from a (B,C,H,W) or (C,H,W) tensor.
    pad_params: (top, bot, left, right)
    Returns tensor cropped back to (B,C,original_size,original_size).
    """
    top, bot, left, right = pad_params
    h = tensor.shape[-2]
    w = tensor.shape[-1]
    return tensor[..., top:h-bot, left:w-right]


def get_dataloader(data_root, split_file, split='train', batch_size=4,
                   num_workers=2, use_skeleton=False):
    """Factory function to build a DataLoader for a given split."""
    split_data = load_split(split_file)
    ids = split_data[split]
    training = (split == 'train')
    dataset = DCA1Dataset(data_root, ids, training=training, use_skeleton=use_skeleton)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
    )
    return loader


def compute_pos_weight(data_root, train_ids):
    """
    Compute positive-class weight = background_pixels / vessel_pixels
    from training masks. Used for BCE loss weighting.
    """
    total_pos = 0
    total_neg = 0
    for image_id in train_ids:
        mask_path = os.path.join(data_root, 'masks', f'{image_id}_gt.pgm')
        mask = np.array(Image.open(mask_path).convert('L'))
        binary = (mask > 0).astype(np.float32)
        total_pos += binary.sum()
        total_neg += (1 - binary).sum()
    pos_weight = total_neg / (total_pos + 1e-6)
    return float(pos_weight)
