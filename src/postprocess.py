"""
Post-processing for binary segmentation predictions.

Operations (identical for M1 and M2 per SRS):
  1. Remove connected components smaller than min_size pixels
  2. Optional light morphological closing

All operations on numpy binary arrays (uint8 or bool).
N (min_component_size) is chosen on the validation set, then frozen for test.
"""
import numpy as np
from scipy import ndimage


def remove_small_components(binary_mask, min_size=50):
    """
    Remove 8-connected components with fewer than min_size pixels.

    Args:
        binary_mask (np.ndarray): bool or uint8 array, shape (H,W)
        min_size (int): minimum component size to keep

    Returns:
        np.ndarray: cleaned binary mask (uint8)
    """
    mask = binary_mask.astype(bool)
    labeled, num_features = ndimage.label(mask, structure=np.ones((3,3), dtype=int))
    if num_features == 0:
        return mask.astype(np.uint8)

    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    keep = np.zeros_like(labeled, dtype=bool)
    for i, size in enumerate(component_sizes, start=1):
        if size >= min_size:
            keep[labeled == i] = True
    return keep.astype(np.uint8)


def morphological_closing(binary_mask, kernel_size=3):
    """
    Light morphological closing to fill small gaps.

    Args:
        binary_mask (np.ndarray): bool or uint8 array
        kernel_size (int): structuring element size (default 3)

    Returns:
        np.ndarray: closed binary mask (uint8)
    """
    struct = ndimage.generate_binary_structure(2, 2)
    if kernel_size > 3:
        struct = ndimage.iterate_structure(struct, kernel_size // 2)
    closed = ndimage.binary_closing(binary_mask.astype(bool), structure=struct)
    return closed.astype(np.uint8)


def postprocess(binary_mask, min_component_size=50, apply_closing=False, closing_size=3):
    """
    Full post-processing pipeline.

    Args:
        binary_mask (np.ndarray): (H,W) binary array
        min_component_size (int): remove components smaller than this
        apply_closing (bool): whether to apply morphological closing
        closing_size (int): kernel size for closing

    Returns:
        np.ndarray: processed binary mask (uint8)
    """
    result = remove_small_components(binary_mask, min_size=min_component_size)
    if apply_closing:
        result = morphological_closing(result, kernel_size=closing_size)
    return result


def choose_min_component_size(val_preds, val_gts, candidates=None):
    """
    Choose min_component_size on validation set by maximizing Dice.
    val_preds: list of (H,W) binary numpy arrays
    val_gts:   list of (H,W) binary numpy arrays
    Returns optimal min_size.
    """
    from src.metrics import dice_score
    if candidates is None:
        candidates = [0, 10, 25, 50, 100, 200]

    best_size, best_dice = 0, -1.0
    for size in candidates:
        dices = []
        for pred, gt in zip(val_preds, val_gts):
            cleaned = remove_small_components(pred, min_size=size)
            dices.append(dice_score(cleaned, gt))
        mean_dice = np.mean(dices)
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_size = size
    print(f"[PostProcess] Optimal min_component_size={best_size}  val_Dice={best_dice:.4f}")
    return best_size
