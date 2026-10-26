"""
Unit tests for src/metrics.py.

Test cases per SRS S1:
  1. Perfect match        → Dice=1, IoU=1, clDice=1, HD95=0
  2. Empty prediction     → Dice=0, clDice=0, handles gracefully (no exception)
  3. All-background pred  → specificity=1, recall=0
  4. Shifted mask (2px)   → boundary_F1 ≈ 1 at tol=2px
  5. Broken line          → Dice high, clDice drops, component count increases
  6. Perfect background   → all-zero pred, all-zero gt → dice=1

Run with: pytest tests/test_metrics.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.metrics import (
    dice_score, iou_score, precision_recall_specificity,
    f1_score, boundary_f1, hd95_asd, cldice_score,
    connected_components_stats, accuracy, mean_class_accuracy,
    per_image_metrics, aggregate_metrics
)


# ============================================================
# Fixtures: synthetic masks
# ============================================================

def make_perfect_pair(size=64):
    """Identical binary masks."""
    mask = np.zeros((size, size), dtype=bool)
    mask[10:20, 10:50] = True  # horizontal bar
    return mask.astype(np.uint8), mask.astype(np.uint8)


def make_horizontal_line(size=64, row=32, start=5, end=59):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[row, start:end] = 1
    return mask


def make_broken_line(size=64, row=32, start=5, end=59, gap_start=25, gap_end=35):
    """Horizontal line with a gap in the middle."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[row, start:gap_start] = 1
    mask[row, gap_end:end] = 1
    return mask


# ============================================================
# Case 1: Perfect match
# ============================================================

class TestPerfectMatch:
    def test_dice(self):
        pred, gt = make_perfect_pair()
        assert dice_score(pred, gt) == pytest.approx(1.0, abs=1e-5)

    def test_iou(self):
        pred, gt = make_perfect_pair()
        assert iou_score(pred, gt) == pytest.approx(1.0, abs=1e-5)

    def test_precision_recall_spec(self):
        pred, gt = make_perfect_pair()
        prec, rec, spec = precision_recall_specificity(pred, gt)
        assert prec == pytest.approx(1.0, abs=1e-4)
        assert rec  == pytest.approx(1.0, abs=1e-4)
        assert spec == pytest.approx(1.0, abs=1e-4)

    def test_hd95_zero(self):
        pred, gt = make_perfect_pair()
        hd95, asd = hd95_asd(pred, gt)
        assert hd95 == pytest.approx(0.0, abs=1e-3)
        assert asd  == pytest.approx(0.0, abs=1e-3)

    def test_cldice_one(self):
        pred, gt = make_perfect_pair()
        tp, tr, cl = cldice_score(pred, gt)
        assert cl == pytest.approx(1.0, abs=1e-3)

    def test_components(self):
        pred, gt = make_perfect_pair()
        n_pred, n_gt, diff = connected_components_stats(pred, gt)
        assert n_pred == n_gt
        assert diff == 0


# ============================================================
# Case 2: Empty prediction
# ============================================================

class TestEmptyPrediction:
    def setup_method(self):
        self.pred = np.zeros((64, 64), dtype=np.uint8)
        self.gt   = make_perfect_pair()[1]

    def test_dice_zero(self):
        assert dice_score(self.pred, self.gt) == pytest.approx(0.0, abs=1e-5)

    def test_no_exception(self):
        """All metrics must handle empty prediction without raising."""
        result = per_image_metrics(self.pred, self.gt)
        assert isinstance(result, dict)

    def test_hd95_empty_pred(self):
        """Empty prediction → image diagonal returned."""
        hd95, asd = hd95_asd(self.pred, self.gt)
        img_diag = float(np.sqrt(64**2 + 64**2))
        assert hd95 == pytest.approx(img_diag, rel=0.01)

    def test_cldice_zero(self):
        _, _, cl = cldice_score(self.pred, self.gt)
        assert cl == pytest.approx(0.0, abs=1e-5)

    def test_recall_zero(self):
        _, rec, _ = precision_recall_specificity(self.pred, self.gt)
        assert rec == pytest.approx(0.0, abs=1e-5)


# ============================================================
# Case 3: All-background prediction (all zeros)
# ============================================================

class TestAllBackground:
    def setup_method(self):
        self.pred = np.zeros((64, 64), dtype=np.uint8)
        self.gt   = np.zeros((64, 64), dtype=np.uint8)

    def test_both_empty_dice(self):
        """Both empty → Dice = 1 (perfect agreement on background)."""
        assert dice_score(self.pred, self.gt) == pytest.approx(1.0, abs=1e-5)

    def test_specificity_one(self):
        gt_with_vessel = make_perfect_pair()[1]
        _, rec, spec = precision_recall_specificity(self.pred, gt_with_vessel)
        assert spec == pytest.approx(1.0, abs=1e-4)
        assert rec  == pytest.approx(0.0, abs=1e-4)


# ============================================================
# Case 4: Shifted mask (2px) → boundary_F1 ≈ 1 at tol=2
# ============================================================

class TestShiftedMask:
    def setup_method(self):
        self.gt = np.zeros((64, 64), dtype=np.uint8)
        self.gt[20:40, 20:40] = 1
        # Shift by 2 pixels
        self.pred = np.zeros((64, 64), dtype=np.uint8)
        self.pred[22:42, 22:42] = 1

    def test_boundary_f1_with_tolerance(self):
        """At tolerance=2px, a 2px-shifted boundary should score close to 1."""
        bf1 = boundary_f1(self.pred, self.gt, tolerance=2)
        assert bf1 > 0.7, f"Expected boundary_F1 > 0.7 for 2px shift with tol=2, got {bf1:.3f}"

    def test_boundary_f1_no_tolerance(self):
        """Without tolerance, a 2px shift lowers boundary F1."""
        bf1_no_tol = boundary_f1(self.pred, self.gt, tolerance=0)
        bf1_tol    = boundary_f1(self.pred, self.gt, tolerance=2)
        assert bf1_tol >= bf1_no_tol


# ============================================================
# Case 5: Broken line
# ============================================================

class TestBrokenLine:
    def setup_method(self):
        """
        GT: continuous horizontal line
        Pred: same line but with a gap in the middle
        → Dice should be high (most pixels correct)
        → clDice should drop (skeleton disconnected)
        → Component count in pred > 1
        """
        self.gt   = make_horizontal_line(64, row=32, start=5, end=59)
        self.pred = make_broken_line(64, row=32, start=5, end=59,
                                      gap_start=25, gap_end=35)

    def test_dice_high(self):
        d = dice_score(self.pred, self.gt)
        # Line is 54px, gap is 10px → ~81% of GT pixels correct
        assert d > 0.6, f"Expected Dice > 0.6 for mostly-correct broken line, got {d:.3f}"

    def test_cldice_drops(self):
        """clDice should be clearly lower than Dice due to broken topology."""
        d  = dice_score(self.pred, self.gt)
        _, _, cl = cldice_score(self.pred, self.gt)
        # clDice should be noticeably lower than Dice
        assert cl < d + 0.1, f"Expected clDice ({cl:.3f}) <= Dice ({d:.3f}) for broken line"

    def test_component_count_increases(self):
        """Prediction should have more connected components than GT."""
        n_pred, n_gt, diff = connected_components_stats(self.pred, self.gt)
        assert n_pred >= 2, f"Expected ≥2 components in broken pred, got {n_pred}"
        assert n_gt == 1, f"Expected 1 component in continuous GT, got {n_gt}"
        assert diff >= 1


# ============================================================
# Case 6: Aggregate metrics
# ============================================================

class TestAggregateMetrics:
    def test_aggregate_empty(self):
        result = aggregate_metrics([])
        assert result == {}

    def test_aggregate_single(self):
        pred, gt = make_perfect_pair()
        m = per_image_metrics(pred, gt)
        agg = aggregate_metrics([m])
        assert 'dice' in agg
        assert agg['dice']['mean'] == pytest.approx(1.0, abs=1e-5)
        assert agg['dice']['std']  == pytest.approx(0.0, abs=1e-5)

    def test_aggregate_multiple(self):
        pred, gt = make_perfect_pair()
        m1 = per_image_metrics(pred, gt)
        pred2 = np.zeros_like(pred)
        m2 = per_image_metrics(pred2, gt)
        agg = aggregate_metrics([m1, m2])
        assert 0.0 < agg['dice']['mean'] < 1.0


# ============================================================
# Shape / dtype safety
# ============================================================

class TestEdgeCases:
    def test_float_input(self):
        """Metrics should work with float arrays (0.0/1.0)."""
        pred = np.array([[0., 1., 1.], [0., 1., 0.]])
        gt   = np.array([[0., 1., 1.], [0., 1., 0.]])
        assert dice_score(pred, gt) == pytest.approx(1.0, abs=1e-5)

    def test_accuracy_perfect(self):
        pred, gt = make_perfect_pair()
        assert accuracy(pred, gt) == pytest.approx(1.0, abs=1e-5)

    def test_mean_class_acc(self):
        pred, gt = make_perfect_pair()
        mca = mean_class_accuracy(pred, gt)
        assert mca == pytest.approx(1.0, abs=1e-4)
