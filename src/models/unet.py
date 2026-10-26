"""
Baseline U-Net for coronary vessel segmentation (M1).

Architecture:
  Encoder: 4 stages of double-conv (3x3, BN, ReLU) + MaxPool
  Bottleneck: double-conv at 512 channels
  Decoder: bilinear upsample + skip concat + double-conv
  Output: 1x1 conv → 1-channel logit

Channel widths: 32-64-128-256-512
Input:  (B, 1, H, W)  — single-channel greyscale, padded to 320x320
Output: (B, 1, H, W)  — logits, same spatial size
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two sequential 3x3 conv + BN + ReLU blocks."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample: MaxPool 2x2 then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample (bilinear) + skip concatenation + DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels = encoder_channels + decoder_channels (after concat)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: decoder feature (upsampled)
        x2: skip connection from encoder
        """
        x1 = self.up(x1)
        # Handle size mismatch (edge case with non-power-of-2 sizes)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        if diff_h != 0 or diff_w != 0:
            x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                             diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard 2D U-Net baseline (M1).
    
    Args:
        in_channels (int): input channels (1 for greyscale)
        out_channels (int): output channels (1 for binary segmentation)
        base_channels (int): first encoder stage channels (default 32)
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels  # 32
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, c)          # 32
        self.enc2 = Down(c,      c * 2)                 # 64
        self.enc3 = Down(c * 2,  c * 4)                 # 128
        self.enc4 = Down(c * 4,  c * 8)                 # 256
        
        # Bottleneck
        self.bottleneck = Down(c * 8, c * 16)           # 512
        
        # Decoder
        self.dec4 = Up(c * 16 + c * 8, c * 8)          # 512+256 → 256
        self.dec3 = Up(c * 8  + c * 4, c * 4)          # 256+128 → 128
        self.dec2 = Up(c * 4  + c * 2, c * 2)          # 128+64  → 64
        self.dec1 = Up(c * 2  + c,     c)               # 64+32   → 32
        
        # Output head
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # (B, 32,  H,    W)
        e2 = self.enc2(e1)         # (B, 64,  H/2,  W/2)
        e3 = self.enc3(e2)         # (B, 128, H/4,  W/4)
        e4 = self.enc4(e3)         # (B, 256, H/8,  W/8)
        
        # Bottleneck
        b  = self.bottleneck(e4)   # (B, 512, H/16, W/16)
        
        # Decoder with skip connections
        d4 = self.dec4(b,  e4)    # (B, 256, H/8,  W/8)
        d3 = self.dec3(d4, e3)    # (B, 128, H/4,  W/4)
        d2 = self.dec2(d3, e2)    # (B, 64,  H/2,  W/2)
        d1 = self.dec1(d2, e1)    # (B, 32,  H,    W)
        
        return self.out_conv(d1)   # (B, 1, H, W)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(config=None):
    """Build U-Net from config dict."""
    base_ch = 32
    if config and 'channels' in config:
        base_ch = config['channels'][0]
    return UNet(in_channels=1, out_channels=1, base_channels=base_ch)
