"""
Efficiency profiling: parameters, FLOPs, training time, inference time, GPU memory.

Usage:
    python -m src.profile --config configs/baseline.yaml
    python -m src.profile --config configs/hybrid_i2.yaml
"""
import os
import yaml
import argparse
import time
import torch
import numpy as np

try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE = True
except ImportError:
    try:
        from thop import profile as thop_profile
        _THOP = True
        _FVCORE = False
    except ImportError:
        _FVCORE = False
        _THOP   = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size=(1, 1, 320, 320), device='cpu'):
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    if _FVCORE:
        flops = FlopCountAnalysis(model, dummy)
        return flops.total()
    elif _THOP:
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return int(macs * 2)   # FLOPs ≈ 2 * MACs
    else:
        return None


def measure_inference_time(model, input_size=(1, 1, 320, 320), device=None,
                            n_warmup=10, n_runs=100):
    """
    Measure per-image inference time (batch size 1).
    Uses torch.cuda.synchronize() for accurate GPU timing.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return float(np.mean(times)) * 1000, float(np.std(times)) * 1000  # ms


def measure_peak_memory(model, input_size=(1, 1, 320, 320), device=None):
    """Measure peak GPU memory during inference (MB)."""
    if device is None or device.type != 'cuda':
        return None
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return peak_mb


def profile_model(config_path):
    from src.models.hybrid_unet import build_model
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    model.eval()

    n_params = count_parameters(model)
    flops    = count_flops(model, device=str(device))
    inf_mean, inf_std = measure_inference_time(model, device=device)
    peak_mem = measure_peak_memory(model, device=device) if device.type == 'cuda' else None

    print(f"\n{'='*50}")
    print(f"Model: {config.get('experiment_name', config_path)}")
    print(f"  Parameters:     {n_params:,}")
    if flops:
        print(f"  FLOPs:          {flops/1e9:.2f} G")
    else:
        print("  FLOPs:          [install fvcore or thop to calculate]")
    print(f"  Inference time: {inf_mean:.2f} ± {inf_std:.2f} ms  (batch=1, 100 runs)")
    if peak_mem:
        print(f"  Peak GPU mem:   {peak_mem:.1f} MB")
    print(f"{'='*50}")

    return {
        'experiment': config.get('experiment_name'),
        'parameters': n_params,
        'flops': flops,
        'inference_mean_ms': inf_mean,
        'inference_std_ms': inf_std,
        'peak_gpu_mb': peak_mem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    profile_model(args.config)


if __name__ == '__main__':
    main()
