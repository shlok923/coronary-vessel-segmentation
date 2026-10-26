"""
Hybrid CNN–Mamba U-Net (M2) for coronary vessel segmentation.

Architecture:
  Stage 1 (highest res):  CNN DoubleConv   — preserves fine vessel detail
  Stage 2:                CNN DoubleConv
  Stage 3:                VSS Block(s)     — long-range context along vessels
  Stage 4 (bottleneck):   VSS Block(s)
  Decoder: skip connections + light CNN/VSS mix

Innovation I2 (use_innovation_i2=True):
  An auxiliary centerline (skeleton) prediction head branches from the decoder.
  This head is trained with a separate BCE loss on skeleton GT.
  The main segmentation loss also includes the clDice term.
  I2 is fully switchable via the config flag.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import DoubleConv, Up
from src.models.vss_block import VSSBlock, log_backend


class VSSStage(nn.Module):
    """
    One encoder/decoder stage using VSS blocks.
    Wraps one or more VSSBlocks with an optional linear channel projection.
    """
    def __init__(self, in_channels, out_channels, num_blocks=1, d_state=16, expand=2):
        super().__init__()
        # Channel adaptation if needed
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                    if in_channels != out_channels else nn.Identity()
        self.blocks = nn.Sequential(*[
            VSSBlock(out_channels, d_state=d_state, expand=expand)
            for _ in range(num_blocks)
        ])
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        return self.norm(x)


class VSSDown(nn.Module):
    """Downsample + VSS stage (analogous to UNet's Down block)."""
    def __init__(self, in_channels, out_channels, num_blocks=1, d_state=16):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.stage = VSSStage(in_channels, out_channels,
                               num_blocks=num_blocks, d_state=d_state)

    def forward(self, x):
        return self.stage(self.pool(x))


class CenterlineHead(nn.Module):
    """
    Auxiliary head for centerline (skeleton) prediction.
    Innovation I2: branches off from the decoder feature map.
    Predicts a binary skeleton mask (same spatial size as main output).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)  # (B, 1, H, W) logits


class HybridUNet(nn.Module):
    """
    Hybrid CNN–Mamba U-Net (M2).
    
    Args:
        in_channels (int): 1 for greyscale
        out_channels (int): 1 for binary segmentation
        channels (list): per-stage channel widths [32, 64, 128, 256, 512]
        cnn_stages (int): number of encoder stages using CNN (from high-res)
        use_innovation_i2 (bool): enable auxiliary centerline head + clDice
        d_state (int): Mamba state dimension
        expand (int): Mamba expansion factor
    """
    def __init__(self, in_channels=1, out_channels=1,
                 channels=None, cnn_stages=2,
                 use_innovation_i2=False,
                 d_state=16, expand=2):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]
        assert len(channels) == 5, "Need exactly 5 channel values [c1,c2,c3,c4,bottleneck]"

        self.use_i2   = use_innovation_i2
        self.channels = channels
        c = channels

        log_backend()

        # ---- Encoder ----
        # Stage 1: CNN (highest resolution, fine detail)
        self.enc1 = DoubleConv(in_channels, c[0])   # CNN

        # Stage 2: CNN or VSS
        if cnn_stages >= 2:
            self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c[0], c[1]))
        else:
            self.enc2 = VSSDown(c[0], c[1], d_state=d_state)

        # Stage 3: VSS (long-range vessel context)
        self.enc3 = VSSDown(c[1], c[2], d_state=d_state)

        # Stage 4 + Bottleneck: VSS
        self.enc4       = VSSDown(c[2], c[3], d_state=d_state)
        self.bottleneck = VSSDown(c[3], c[4], d_state=d_state)

        # ---- Decoder ----
        # Use same Up blocks as U-Net (bilinear + skip concat + DoubleConv)
        self.dec4 = Up(c[4] + c[3], c[3])
        self.dec3 = Up(c[3] + c[2], c[2])
        self.dec2 = Up(c[2] + c[1], c[1])
        self.dec1 = Up(c[1] + c[0], c[0])

        # ---- Output head ----
        self.out_conv = nn.Conv2d(c[0], out_channels, kernel_size=1)

        # ---- Innovation I2: auxiliary centerline head ----
        if self.use_i2:
            self.centerline_head = CenterlineHead(c[0])
            print("[HybridUNet] Innovation I2 ENABLED: centerline auxiliary head active")
        else:
            self.centerline_head = None
            print("[HybridUNet] Innovation I2 disabled")

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) — padded to 320x320

        Returns:
            If use_i2=False: logits (B, 1, H, W)
            If use_i2=True:  (logits, skel_logits) both (B, 1, H, W)
        """
        # Encoder
        e1 = self.enc1(x)           # (B, c0, H,    W)
        e2 = self.enc2(e1)          # (B, c1, H/2,  W/2)
        e3 = self.enc3(e2)          # (B, c2, H/4,  W/4)
        e4 = self.enc4(e3)          # (B, c3, H/8,  W/8)
        b  = self.bottleneck(e4)    # (B, c4, H/16, W/16)

        # Decoder
        d4 = self.dec4(b,  e4)     # (B, c3, H/8,  W/8)
        d3 = self.dec3(d4, e3)     # (B, c2, H/4,  W/4)
        d2 = self.dec2(d3, e2)     # (B, c1, H/2,  W/2)
        d1 = self.dec1(d2, e1)     # (B, c0, H,    W)

        logits = self.out_conv(d1)  # (B, 1, H, W)

        if self.use_i2 and self.centerline_head is not None:
            skel_logits = self.centerline_head(d1)  # (B, 1, H, W)
            return logits, skel_logits

        return logits

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_hybrid_unet(config):
    """Build HybridUNet from config dict."""
    return HybridUNet(
        in_channels=1,
        out_channels=1,
        channels=config.get('channels', [32, 64, 128, 256, 512]),
        cnn_stages=config.get('cnn_stages', 2),
        use_innovation_i2=config.get('use_innovation_i2', False),
        d_state=config.get('d_state', 16),
        expand=config.get('expand', 2),
    )


def build_model(config):
    """Top-level model factory. Dispatches on config['model']."""
    from src.models.unet import build_unet
    model_name = config.get('model', 'unet')
    if model_name == 'unet':
        return build_unet(config)
    elif model_name == 'hybrid_unet':
        return build_hybrid_unet(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
