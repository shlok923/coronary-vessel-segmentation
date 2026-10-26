"""
Visual State Space (VSS) Block with 2D Selective Scan (SS2D).

Architecture per block:
  Input → LayerNorm → Linear projection → Depthwise conv → SiLU
        → SS2D (4-direction selective scan) → merge → gated output
        → Residual add → Output

Backend auto-selection:
  1. Try mamba-ssm CUDA kernels (fast, requires CUDA + mamba-ssm installed)
  2. Fall back to pure-PyTorch selective scan (slower but always works)

Attribution:
  Adapted from VM-UNet (Ruan & Xiang, 2024):
    https://github.com/JCruan519/VM-UNet
  and VMamba (Liu et al., 2024):
    https://github.com/MzeroMiko/VMamba
  Used under Apache 2.0 license. Core SS2D idea retained; code restructured
  and simplified for 300x300 coronary angiogram segmentation.
"""
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_MAMBA_AVAILABLE = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_AVAILABLE = True
    logger.info("[VSS] Backend: mamba-ssm CUDA kernels (fast)")
except ImportError:
    logger.info("[VSS] Backend: pure-PyTorch selective scan fallback (slower)")


# ---------------------------------------------------------------------------
# Pure-PyTorch selective scan (fallback)
# ---------------------------------------------------------------------------
def selective_scan_pytorch(u, delta, A, B, C, D, delta_bias=None, delta_softplus=True):
    """
    Pure-PyTorch implementation of selective scan (S4/Mamba recurrence).
    
    Args:
        u:     (B, L, D) input
        delta: (B, L, D) dt
        A:     (D, N) state matrix (log-space)
        B:     (B, L, N) input projection
        C:     (B, L, N) output projection
        D:     (D,) skip connection
        
    Returns:
        y: (B, L, D) output
    """
    B_batch, L, D = u.shape
    N = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        delta = F.softplus(delta)

    # Discretize: A_bar = exp(delta * A), B_bar = delta * B
    delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,D,N)
    delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)      # (B,L,D,N)

    # Scan
    x = torch.zeros(B_batch, D, N, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(L):
        x = delta_A[:, i] * x + delta_B_u[:, i]   # (B,D,N)
        y = (x * C[:, i].unsqueeze(1)).sum(-1)      # (B,D)
        ys.append(y)
    y = torch.stack(ys, dim=1)                       # (B,L,D)
    y = y + u * D.unsqueeze(0).unsqueeze(0)
    return y


# ---------------------------------------------------------------------------
# SS2D: 2D multi-directional selective scan
# ---------------------------------------------------------------------------
class SS2D(nn.Module):
    """
    2D Selective Scan module. Scans the 2D feature map in 4 directions:
      1. Row-major (left→right, top→bottom)
      2. Reverse row-major
      3. Column-major (top→bottom, left→right)
      4. Reverse column-major
    
    Then merges all 4 scan outputs.
    """
    def __init__(self, d_model, d_state=16, expand=2, dt_rank='auto',
                 dt_min=0.001, dt_max=0.1, dt_init='random',
                 dt_scale=1.0, dt_init_floor=1e-4):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        K = 4  # number of scan directions

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d  = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3,
                                  padding=1, groups=self.d_inner, bias=True)
        self.act     = nn.SiLU()

        # SSM parameters (shared across directions would reduce params;
        # independent is more expressive — we use independent here)
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            for _ in range(K)
        ])
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(K)
        ])

        # Initialize dt projections
        for dt_proj in self.dt_projs:
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)

        # A matrix (log-space, diagonal)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, -1)  # (d_inner, d_state)
        self.A_logs = nn.ParameterList([
            nn.Parameter(torch.log(A.clone())) for _ in range(K)
        ])

        # D (skip connection)
        self.Ds = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner)) for _ in range(K)
        ])

        # Output normalization and projection
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _scan_direction(self, x_flat, direction_idx):
        """Run selective scan for one direction."""
        B, L, D = x_flat.shape
        xz = self.x_proj[direction_idx](x_flat)  # (B,L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(
            xz, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_projs[direction_idx](dt)      # (B,L, d_inner)
        A  = -torch.exp(self.A_logs[direction_idx].float())  # (d_inner, d_state)
        D  = self.Ds[direction_idx].float()

        if _MAMBA_AVAILABLE:
            # Use fast CUDA kernel
            # selective_scan_fn expects (B, D, L) layout
            y = selective_scan_fn(
                x_flat.permute(0,2,1).contiguous(),
                dt.permute(0,2,1).contiguous(),
                A,
                B_ssm.permute(0,2,1).contiguous(),
                C_ssm.permute(0,2,1).contiguous(),
                D,
                delta_softplus=True,
            ).permute(0,2,1)  # → (B,L,D)
        else:
            y = selective_scan_pytorch(x_flat, dt, A, B_ssm, C_ssm, D,
                                        delta_softplus=True)
        return y

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) — channels-last format

        Returns:
            out: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Project input
        xz = self.in_proj(x)                          # (B,H,W, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                 # each (B,H,W, d_inner)

        # Depthwise conv + activation
        x_conv = x_in.permute(0,3,1,2).contiguous()   # (B, d_inner, H, W)
        x_conv = self.act(self.conv2d(x_conv))
        x_conv = x_conv.permute(0,2,3,1)              # (B, H, W, d_inner)

        # 4-direction scans
        L = H * W
        # Direction 1: row-major
        x1 = x_conv.reshape(B, L, self.d_inner)
        # Direction 2: reverse row-major
        x2 = x1.flip(1)
        # Direction 3: column-major
        x3 = x_conv.permute(0,2,1,3).reshape(B, L, self.d_inner)
        # Direction 4: reverse column-major
        x4 = x3.flip(1)

        y1 = self._scan_direction(x1, 0)
        y2 = self._scan_direction(x2, 1).flip(1)
        y3 = self._scan_direction(x3, 2).reshape(B, W, H, self.d_inner).permute(0,2,1,3).reshape(B, L, self.d_inner)
        y4 = self._scan_direction(x4, 3).flip(1).reshape(B, W, H, self.d_inner).permute(0,2,1,3).reshape(B, L, self.d_inner)

        # Merge
        y = y1 + y2 + y3 + y4
        y = y.reshape(B, H, W, self.d_inner)
        y = self.out_norm(y)

        # Gated output
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# VSS Block (full block with residual)
# ---------------------------------------------------------------------------
class VSSBlock(nn.Module):
    """
    Visual State Space Block.
    
    Structure:
      Input → LayerNorm → SS2D → Residual → Output
    
    Works with channels-last (B,H,W,C) internally.
    Accepts and returns (B,C,H,W) tensors externally.
    """
    def __init__(self, channels, d_state=16, expand=2, drop_path=0.0):
        super().__init__()
        self.norm  = nn.LayerNorm(channels)
        self.ss2d  = SS2D(d_model=channels, d_state=d_state, expand=expand)
        # Optional stochastic depth
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)

    def forward(self, x):
        """
        Args:  x: (B, C, H, W)
        Returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Convert to channels-last
        x_cl = x.permute(0, 2, 3, 1)              # (B,H,W,C)
        # VSS with residual
        out  = x_cl + self.drop_path(self.ss2d(self.norm(x_cl)))
        # Convert back to channels-first
        return out.permute(0, 3, 1, 2)             # (B,C,H,W)


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand = torch.floor(rand + keep_prob)
        return x / keep_prob * rand


def log_backend():
    """Log which SS2D backend is active. Call once at startup."""
    if _MAMBA_AVAILABLE:
        logger.info("SS2D backend: mamba-ssm CUDA kernels")
        print("[VSS] Using mamba-ssm CUDA backend (fast)")
    else:
        logger.info("SS2D backend: pure-PyTorch fallback")
        print("[VSS] Using pure-PyTorch SS2D fallback (slower, but functional)")
