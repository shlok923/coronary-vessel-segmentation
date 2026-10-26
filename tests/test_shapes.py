"""
Shape unit tests for U-Net and Hybrid U-Net models.

Verifies:
  - Correct input/output shapes
  - Innovation I2 output structure
  - Models run on CPU without errors

Run with: pytest tests/test_shapes.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.models.unet import UNet
from src.models.hybrid_unet import HybridUNet


@pytest.fixture
def dummy_input():
    """Standard (B,1,320,320) tensor — matches padded DCA1 input."""
    return torch.randn(2, 1, 320, 320)


@pytest.fixture
def small_input():
    """Small (B,1,64,64) tensor for quick tests."""
    return torch.randn(2, 1, 64, 64)


# ============================================================
# U-Net shapes
# ============================================================

class TestUNetShapes:
    def test_output_shape_full(self, dummy_input):
        model = UNet(in_channels=1, out_channels=1, base_channels=16)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == dummy_input.shape, \
            f"Expected {dummy_input.shape}, got {out.shape}"

    def test_output_shape_small(self, small_input):
        model = UNet(in_channels=1, out_channels=1, base_channels=16)
        model.eval()
        with torch.no_grad():
            out = model(small_input)
        assert out.shape == small_input.shape

    def test_parameter_count(self):
        model = UNet(base_channels=32)
        n = model.get_num_parameters()
        assert n > 0
        print(f"\nU-Net parameters: {n:,}")

    def test_batch_size_1(self):
        x = torch.randn(1, 1, 320, 320)
        model = UNet(base_channels=16)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 320, 320)


# ============================================================
# Hybrid U-Net shapes
# ============================================================

class TestHybridUNetShapes:
    def test_output_shape_no_i2(self, small_input):
        model = HybridUNet(
            in_channels=1, out_channels=1,
            channels=[16, 32, 64, 128, 256],
            cnn_stages=2, use_innovation_i2=False
        )
        model.eval()
        with torch.no_grad():
            out = model(small_input)
        assert not isinstance(out, tuple), "Without I2, output should be a single tensor"
        assert out.shape == small_input.shape

    def test_output_shape_with_i2(self, small_input):
        model = HybridUNet(
            in_channels=1, out_channels=1,
            channels=[16, 32, 64, 128, 256],
            cnn_stages=2, use_innovation_i2=True
        )
        model.eval()
        with torch.no_grad():
            out = model(small_input)
        assert isinstance(out, tuple), "With I2, output should be (logits, skel_logits)"
        logits, skel_logits = out
        assert logits.shape == small_input.shape
        assert skel_logits.shape == small_input.shape

    def test_parameter_count(self):
        model = HybridUNet(channels=[32, 64, 128, 256, 512])
        n = model.get_num_parameters()
        assert n > 0
        print(f"\nHybrid U-Net parameters: {n:,}")

    def test_all_cnn_stages(self, small_input):
        """Setting cnn_stages=4 should still work."""
        model = HybridUNet(
            channels=[16, 32, 64, 128, 256],
            cnn_stages=4, use_innovation_i2=False
        )
        model.eval()
        with torch.no_grad():
            out = model(small_input)
        assert out.shape == small_input.shape

    def test_single_batch(self):
        x = torch.randn(1, 1, 64, 64)
        model = HybridUNet(channels=[16, 32, 64, 128, 256], use_innovation_i2=False)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 64, 64)

