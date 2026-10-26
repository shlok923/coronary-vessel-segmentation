"""
Loss functions for coronary vessel segmentation.

Available losses:
  - BCEDiceLoss:        standard BCE-with-logits + soft Dice
  - SoftDiceLoss:       standalone soft Dice
  - SoftSkeletonLoss:   soft clDice loss (topology-aware)
  - CombinedLoss:       BCE + Dice + clDice + auxiliary skeleton BCE  (M2+I2)

All losses accept logits (not sigmoid-activated) for the main segmentation head,
and probabilities or logits for auxiliary heads (configurable).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice loss. Input: logits. Target: binary float tensor."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE-with-logits + soft Dice.
    pos_weight: tensor scalar for class imbalance (background/vessel ratio).
    """
    def __init__(self, pos_weight=None, dice_weight=1.0, bce_weight=1.0, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = SoftDiceLoss(smooth=smooth)
        self.pos_weight  = pos_weight  # will be set after computing from data

    def forward(self, logits, targets):
        pw = self.pos_weight
        if pw is not None and not isinstance(pw, torch.Tensor):
            pw = torch.tensor([pw], device=logits.device, dtype=logits.dtype)
        bce  = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def soft_skeletonize(mask, iterations=5):
    """
    Soft (differentiable) morphological thinning approximation.
    Uses repeated erosion via max-pooling to approximate skeleton.
    mask: (B,1,H,W) float tensor in [0,1].
    Returns a soft skeleton approximation.
    """
    # Differentiable erosion: min in neighborhood = -maxpool(-x)
    skel = mask.clone()
    prev = torch.zeros_like(skel)
    for _ in range(iterations):
        eroded = -F.max_pool2d(-skel, kernel_size=3, stride=1, padding=1)
        # Soft skeleton: pixels that are in mask but not in eroded
        prev   = skel - eroded
        skel   = eroded
    return torch.clamp(prev, 0, 1)


class SoftClDiceLoss(nn.Module):
    """
    Soft centerline Dice (clDice) loss.
    Encourages topology-preserving predictions.
    Reference: Shit et al. (2021) - clDice: A Novel Topology-Preserving Loss.

    topo_precision = |skel_pred ∩ mask_gt| / |skel_pred|
    topo_recall    = |skel_gt ∩ mask_pred| / |skel_gt|
    clDice         = 2 * tp * tr / (tp + tr)
    """
    def __init__(self, smooth=1.0, skel_iterations=5):
        super().__init__()
        self.smooth = smooth
        self.skel_iterations = skel_iterations

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # Soft skeleton of prediction and ground truth
        skel_pred = soft_skeletonize(probs,   self.skel_iterations)
        skel_gt   = soft_skeletonize(targets, self.skel_iterations)

        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)
        sp = skel_pred.view(skel_pred.size(0), -1)
        sg = skel_gt.view(skel_gt.size(0), -1)

        topo_prec = (sp * t).sum(1) / (sp.sum(1) + self.smooth)
        topo_rec  = (sg * p).sum(1) / (sg.sum(1) + self.smooth)
        cl_dice   = 2.0 * topo_prec * topo_rec / (topo_prec + topo_rec + self.smooth)
        return 1.0 - cl_dice.mean()


class CombinedLoss(nn.Module):
    """
    Full loss for M2+I2:
      L = bce_weight * BCE + dice_weight * Dice
        + cldice_weight * clDice
        + skel_weight * BCE(aux_skel_logits, skel_gt)

    If aux_skel_logits is None (innovation disabled), skel_weight term is skipped.
    """
    def __init__(self, pos_weight=None, bce_weight=1.0, dice_weight=1.0,
                 cldice_weight=0.5, skel_weight=0.3, smooth=1.0):
        super().__init__()
        self.bce_dice   = BCEDiceLoss(pos_weight=pos_weight,
                                       dice_weight=dice_weight,
                                       bce_weight=bce_weight,
                                       smooth=smooth)
        self.cldice     = SoftClDiceLoss(smooth=smooth)
        self.cldice_weight = cldice_weight
        self.skel_weight   = skel_weight

    def forward(self, logits, targets, aux_skel_logits=None, skel_gt=None):
        loss = self.bce_dice(logits, targets)
        loss = loss + self.cldice_weight * self.cldice(logits, targets)
        if aux_skel_logits is not None and skel_gt is not None:
            skel_bce = F.binary_cross_entropy_with_logits(
                aux_skel_logits, skel_gt
            )
            loss = loss + self.skel_weight * skel_bce
        return loss


def build_loss(config, pos_weight=None):
    """
    Build the appropriate loss from config dict.
    config keys: loss (str), cldice_weight, skel_weight, bce_weight, dice_weight
    """
    loss_name = config.get('loss', 'bce_dice')
    pw = pos_weight if config.get('pos_weight') == 'auto' else config.get('pos_weight')

    if loss_name == 'bce_dice':
        return BCEDiceLoss(
            pos_weight=pw,
            bce_weight=config.get('bce_weight', 1.0),
            dice_weight=config.get('dice_weight', 1.0),
        )
    elif loss_name == 'bce_dice_cldice':
        return CombinedLoss(
            pos_weight=pw,
            bce_weight=config.get('bce_weight', 1.0),
            dice_weight=config.get('dice_weight', 1.0),
            cldice_weight=config.get('cldice_weight', 0.5),
            skel_weight=config.get('skel_weight', 0.3),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
