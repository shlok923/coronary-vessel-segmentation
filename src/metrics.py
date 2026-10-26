"""
Evaluation metrics for binary coronary vessel segmentation.

All metrics operate on numpy arrays (H,W) with values in {0,1}.
Probabilities (float [0,1]) are accepted where noted (for AUC metrics).

Metrics implemented:
  - dice_score, iou_score
  - precision, recall, specificity
  - f1_score (alias of dice for completeness)
  - boundary_f1 (tolerance-based, tol=2px)
  - hd95, asd (Hausdorff 95th percentile and Average Surface Distance)
  - cldice (centerline Dice: topo_precision, topo_recall, harmonic mean)
  - connected_components_stats
  - accuracy, mean_class_accuracy
  - auc_roc, auc_pr
  - thin_thick_vessel_recall (width threshold)
  - per_image_metrics (runs all metrics for one image)
  - aggregate_metrics (mean ± std over a list of per-image results)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-7


# ============================================================
# Basic overlap metrics
# ============================================================

def dice_score(pred, gt):
    """
    Dice coefficient (F1).
    pred, gt: (H,W) binary numpy arrays.
    Returns float in [0,1].
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty → perfect
    return float(2.0 * intersection / denom)


def iou_score(pred, gt):
    """Intersection over Union (Jaccard)."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def precision_recall_specificity(pred, gt):
    """
    Returns (precision, recall, specificity) as floats.
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)   [sensitivity]
    specificity = TN / (TN + FP)
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    TP = (pred &  gt).sum()
    FP = (pred & ~gt).sum()
    FN = (~pred &  gt).sum()
    TN = (~pred & ~gt).sum()
    prec = float(TP / (TP + FP + EPS))
    rec  = float(TP / (TP + FN + EPS))
    spec = float(TN / (TN + FP + EPS))
    return prec, rec, spec


def f1_score(pred, gt):
    """F1 = Dice (identical for binary masks). Reported separately per SRS."""
    return dice_score(pred, gt)


def accuracy(pred, gt):
    """Pixel-wise accuracy = (TP + TN) / total."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    correct = (pred == gt).sum()
    return float(correct / pred.size)


def mean_class_accuracy(pred, gt):
    """Mean of sensitivity and specificity."""
    _, rec, spec = precision_recall_specificity(pred, gt)
    return float((rec + spec) / 2.0)


# ============================================================
# Boundary metrics
# ============================================================

def _get_boundary(mask, dilation=1):
    """Extract boundary pixels of a binary mask via erosion."""
    mask = mask.astype(bool)
    struct = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=dilation)
    return mask & ~eroded


def boundary_f1(pred, gt, tolerance=2):
    """
    Tolerance-based boundary F1.
    A predicted boundary pixel is TP if it is within `tolerance` pixels
    of a GT boundary pixel, and vice versa.
    This is distinct from Dice (topology-insensitive) per SRS.
    """
    pred_boundary = _get_boundary(pred.astype(bool))
    gt_boundary   = _get_boundary(gt.astype(bool))

    if gt_boundary.sum() == 0 and pred_boundary.sum() == 0:
        return 1.0
    if gt_boundary.sum() == 0 or pred_boundary.sum() == 0:
        return 0.0

    # Dilate boundaries by tolerance radius
    struct = ndimage.generate_binary_structure(2, 2)
    pred_dilated = ndimage.binary_dilation(pred_boundary, structure=struct,
                                            iterations=tolerance)
    gt_dilated   = ndimage.binary_dilation(gt_boundary,   structure=struct,
                                            iterations=tolerance)

    precision_b = float((pred_boundary & gt_dilated).sum() / (pred_boundary.sum() + EPS))
    recall_b    = float((gt_boundary & pred_dilated).sum() / (gt_boundary.sum() + EPS))
    if precision_b + recall_b < EPS:
        return 0.0
    return float(2.0 * precision_b * recall_b / (precision_b + recall_b + EPS))


# ============================================================
# Distance metrics: HD95 and ASD
# ============================================================

_EMPTY_STRATEGY = 'image_diagonal'  # or 'skip' — documented choice

def hd95_asd(pred, gt):
    """
    Hausdorff distance (95th percentile) and Average Surface Distance.
    Both are symmetric, computed on mask boundaries.
    
    Empty prediction/GT policy:
      If either boundary is empty, return (image_diagonal, image_diagonal).
      This choice is documented and consistent for both models.
    
    Returns: (hd95, asd) in pixels
    """
    H, W = pred.shape
    img_diag = float(np.sqrt(H**2 + W**2))

    pred_b = _get_boundary(pred.astype(bool))
    gt_b   = _get_boundary(gt.astype(bool))

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        # Empty boundary: assign image diagonal
        return img_diag, img_diag

    # Distance transform of each boundary
    pred_b_coords = np.argwhere(pred_b)
    gt_b_coords   = np.argwhere(gt_b)

    # DT from pred boundary to GT boundary (and vice versa)
    dt_pred = distance_transform_edt(~gt_b)   # dist of each pixel to GT boundary
    dt_gt   = distance_transform_edt(~pred_b) # dist of each pixel to pred boundary

    d_pred_to_gt = dt_pred[pred_b]  # distances from pred boundary pixels to GT boundary
    d_gt_to_pred = dt_gt[gt_b]      # distances from GT boundary pixels to pred boundary

    hd95_val = float(max(np.percentile(d_pred_to_gt, 95),
                         np.percentile(d_gt_to_pred, 95)))
    asd_val  = float((d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0)

    return hd95_val, asd_val


# ============================================================
# Topology metrics: clDice
# ============================================================

def cldice_score(pred, gt):
    """
    Centerline Dice (clDice).
    Uses skimage.morphology.skeletonize to extract vessel centerlines.
    
    topo_precision = |skel(pred) ∩ gt|   / |skel(pred)|
    topo_recall    = |skel(gt)  ∩ pred|  / |skel(gt)|
    clDice = 2 * tp * tr / (tp + tr)
    
    Returns: (topo_precision, topo_recall, cldice) as floats
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0, 0.0

    skel_pred = skeletonize(pred)
    skel_gt   = skeletonize(gt)

    tp_prec = float((skel_pred & gt).sum() / (skel_pred.sum() + EPS))
    tp_rec  = float((skel_gt  & pred).sum() / (skel_gt.sum()  + EPS))

    if tp_prec + tp_rec < EPS:
        return tp_prec, tp_rec, 0.0

    cl_dice = float(2.0 * tp_prec * tp_rec / (tp_prec + tp_rec + EPS))
    return tp_prec, tp_rec, cl_dice


# ============================================================
# Connected components
# ============================================================

def connected_components_stats(pred, gt):
    """
    Count 8-connected components in prediction and ground truth.
    Returns: (n_pred, n_gt, abs_diff)
    """
    struct = np.ones((3,3), dtype=int)
    _, n_pred = ndimage.label(pred.astype(bool), structure=struct)
    _, n_gt   = ndimage.label(gt.astype(bool),   structure=struct)
    return n_pred, n_gt, abs(n_pred - n_gt)


# ============================================================
# AUC metrics (require probability maps)
# ============================================================

def auc_roc_pr(prob_map, gt):
    """
    AUC-ROC and AUC-PR from raw probability map [0,1].
    prob_map: (H,W) float array
    gt:       (H,W) binary array
    Returns: (auc_roc, auc_pr)
    """
    gt_flat   = gt.astype(int).ravel()
    prob_flat = prob_map.ravel()

    # Need both classes present
    if gt_flat.sum() == 0 or (1 - gt_flat).sum() == 0:
        return float('nan'), float('nan')

    try:
        auc_roc = roc_auc_score(gt_flat, prob_flat)
        auc_pr  = average_precision_score(gt_flat, prob_flat)
    except Exception:
        return float('nan'), float('nan')

    return float(auc_roc), float(auc_pr)


# ============================================================
# Thin vs Thick vessel recall
# ============================================================

def thin_thick_vessel_recall(pred, gt, thin_threshold=3):
    """
    Compute recall for thin (width ≤ thin_threshold px) and thick vessels.
    
    Method:
      1. Skeletonize GT mask
      2. On skeleton pixels, compute local vessel width = 2 * distance_transform(GT)
      3. Classify skeleton pixels as thin (width ≤ threshold) or thick (> threshold)
      4. Propagate class labels to all vessel pixels via nearest-neighbor on skeleton
      5. Compute recall separately for each class
    
    Args:
        pred:           (H,W) binary array
        gt:             (H,W) binary array
        thin_threshold: width in pixels to classify thin vs thick (default 3)
    
    Returns:
        dict with keys: thin_recall, thick_recall, thin_fraction, thick_fraction,
                        width_values (np.array of widths at skeleton pixels)
    """
    gt   = gt.astype(bool)
    pred = pred.astype(bool)

    if gt.sum() == 0:
        return dict(thin_recall=float('nan'), thick_recall=float('nan'),
                    thin_fraction=float('nan'), thick_fraction=float('nan'),
                    width_values=np.array([]))

    # Distance transform of GT mask → gives radius at each pixel
    dt = distance_transform_edt(gt)
    skel = skeletonize(gt)

    if skel.sum() == 0:
        return dict(thin_recall=float('nan'), thick_recall=float('nan'),
                    thin_fraction=float('nan'), thick_fraction=float('nan'),
                    width_values=np.array([]))

    # Width at skeleton pixels = 2 * radius
    skel_coords = np.argwhere(skel)
    widths = 2.0 * dt[skel[:, None] if skel.ndim == 1 else skel]
    # Properly index:
    widths = 2.0 * dt[skel]

    # Classify skeleton pixels
    skel_labels = np.where(widths <= thin_threshold, 'thin', 'thick')

    # Build a width map: for each GT pixel, find nearest skeleton pixel
    # Use DT on ~skel to get nearest skeleton pixel index
    skel_dt, skel_idx = distance_transform_edt(~skel, return_indices=True)
    # For each GT pixel, look up the width at the nearest skeleton pixel
    rows_gt, cols_gt = np.where(gt)
    nearest_skel_row = skel_idx[0][rows_gt, cols_gt]
    nearest_skel_col = skel_idx[1][rows_gt, cols_gt]
    nearest_widths = 2.0 * dt[nearest_skel_row, nearest_skel_col]

    # Classify GT pixels
    thin_mask  = np.zeros_like(gt)
    thick_mask = np.zeros_like(gt)
    for i, (r, c) in enumerate(zip(rows_gt, cols_gt)):
        if nearest_widths[i] <= thin_threshold:
            thin_mask[r, c] = True
        else:
            thick_mask[r, c] = True

    thin_gt  = gt & thin_mask
    thick_gt = gt & thick_mask

    thin_recall  = float((pred & thin_gt).sum()  / (thin_gt.sum()  + EPS))
    thick_recall = float((pred & thick_gt).sum() / (thick_gt.sum() + EPS))

    thin_frac  = float(thin_gt.sum()  / (gt.sum() + EPS))
    thick_frac = float(thick_gt.sum() / (gt.sum() + EPS))

    return dict(
        thin_recall=thin_recall,
        thick_recall=thick_recall,
        thin_fraction=thin_frac,
        thick_fraction=thick_frac,
        width_values=widths,
    )


# ============================================================
# Per-image and aggregate functions
# ============================================================

def per_image_metrics(pred, gt, prob_map=None, thin_threshold=3, boundary_tol=2):
    """
    Compute all metrics for a single image pair.
    
    Args:
        pred:      (H,W) binary array
        gt:        (H,W) binary array
        prob_map:  (H,W) float [0,1] — required for AUC metrics
        thin_threshold: pixel width threshold for thin/thick vessels
        boundary_tol:   pixel tolerance for boundary F1
    
    Returns:
        dict of all metric values
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    prec, rec, spec = precision_recall_specificity(pred, gt)
    hd95_v, asd_v   = hd95_asd(pred, gt)
    topo_prec, topo_rec, cl_dice = cldice_score(pred, gt)
    n_pred, n_gt, cc_diff = connected_components_stats(pred, gt)
    thin_thick = thin_thick_vessel_recall(pred, gt, thin_threshold=thin_threshold)

    m = {
        # Core
        'dice':           dice_score(pred, gt),
        'iou':            iou_score(pred, gt),
        'precision':      prec,
        'recall':         rec,
        'specificity':    spec,
        'f1':             dice_score(pred, gt),         # identical to dice; kept separate
        'boundary_f1':    boundary_f1(pred, gt, tolerance=boundary_tol),
        'hd95':           hd95_v,
        'asd':            asd_v,
        # Topology
        'cldice':         cl_dice,
        'topo_precision': topo_prec,
        'topo_recall':    topo_rec,
        # Components
        'n_components_pred': n_pred,
        'n_components_gt':   n_gt,
        'component_diff':    cc_diff,
        # Additional
        'accuracy':          accuracy(pred, gt),
        'mean_class_acc':    mean_class_accuracy(pred, gt),
        # Thin/thick
        'thin_recall':       thin_thick['thin_recall'],
        'thick_recall':      thin_thick['thick_recall'],
        'thin_fraction':     thin_thick['thin_fraction'],
    }

    if prob_map is not None:
        auc_roc_v, auc_pr_v = auc_roc_pr(prob_map, gt)
        m['auc_roc'] = auc_roc_v
        m['auc_pr']  = auc_pr_v
    else:
        m['auc_roc'] = float('nan')
        m['auc_pr']  = float('nan')

    return m


def aggregate_metrics(per_image_list):
    """
    Aggregate a list of per-image metric dicts into mean ± std.
    
    Returns:
        dict: {metric_name: {'mean': float, 'std': float}}
    """
    if not per_image_list:
        return {}
    keys = [k for k in per_image_list[0].keys()
            if isinstance(per_image_list[0][k], (int, float))]
    result = {}
    for k in keys:
        vals = [m[k] for m in per_image_list if not np.isnan(m.get(k, float('nan')))]
        if vals:
            result[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        else:
            result[k] = {'mean': float('nan'), 'std': float('nan')}
    return result
