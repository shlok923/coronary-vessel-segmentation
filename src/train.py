"""
Training script for coronary vessel segmentation.

Usage:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/hybrid_i2.yaml

Features:
- Fixed seeds for reproducibility
- TensorBoard + CSV logging
- Early stopping on val Dice (or val clDice)
- Best and last checkpoint saving
- Config saved to run directory
- Mixed precision (AMP) support
- Gradient clipping
- AdamW + cosine/ReduceLROnPlateau scheduler
"""
import os, sys, json, yaml, argparse, time, random, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from src.datasets import get_dataloader, compute_pos_weight, load_split
from src.losses import build_loss
from src.metrics import per_image_metrics, aggregate_metrics
from src.models.hybrid_unet import build_model
from src.postprocess import postprocess


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


def get_optimizer(model, config):
    lr  = config.get('lr', 1e-4)
    wd  = config.get('weight_decay', 1e-4)
    opt = config.get('optimizer', 'adamw').lower()
    if opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


def get_scheduler(optimizer, config, steps_per_epoch=None):
    sched = config.get('scheduler', 'cosine').lower()
    epochs = config.get('epochs', 100)
    if sched == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {sched}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0.0
    use_i2 = config.get('use_innovation_i2', False)

    for batch in loader:
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)
        skels  = batch.get('skeleton', None)
        if skels is not None:
            skels = skels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=(device.type == 'cuda')):
            if use_i2:
                logits, skel_logits = model(images)
                loss = criterion(logits, masks,
                                  aux_skel_logits=skel_logits, skel_gt=skels)
            else:
                logits = model(images)
                loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        config.get('grad_clip', 1.0))
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, config, threshold=0.5):
    model.eval()
    total_loss = 0.0
    per_image_results = []
    use_i2 = config.get('use_innovation_i2', False)

    for batch in loader:
        images    = batch['image'].to(device)
        masks     = batch['mask'].to(device)
        pad_params = batch['pad_params']
        skels      = batch.get('skeleton', None)
        if skels is not None:
            skels = skels.to(device)

        with autocast(enabled=(device.type == 'cuda')):
            if use_i2:
                logits, skel_logits = model(images)
                loss = criterion(logits, masks,
                                  aux_skel_logits=skel_logits, skel_gt=skels)
            else:
                logits = model(images)
                loss = criterion(logits, masks)

        total_loss += loss.item()

        # Per-image metrics (crop back to 300x300)
        probs = torch.sigmoid(logits).cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i in range(images.shape[0]):
            top, bot, left, right = [p[i].item() if torch.is_tensor(p[i]) else p[i]
                                       for p in pad_params]
            H, W = probs.shape[2], probs.shape[3]
            prob_crop = probs[i, 0, top:H-bot, left:W-right]
            mask_crop = masks_np[i, 0, top:H-bot, left:W-right]
            pred_crop = (prob_crop > threshold).astype(np.uint8)
            m = per_image_metrics(pred_crop, mask_crop.astype(np.uint8),
                                   prob_map=prob_crop)
            per_image_results.append(m)

    agg = aggregate_metrics(per_image_results)
    val_loss = total_loss / len(loader)
    return val_loss, agg


def train(config, config_path):
    # Setup
    seed = config.get('seed', 42)
    set_seed(seed)

    run_dir = config.get('log_dir', 'results/runs/default')
    ckpt_dir = config.get('checkpoint_dir', 'results/checkpoints/default')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(config, run_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")
    print(f"[Train] Experiment: {config.get('experiment_name', 'unnamed')}")

    # Data
    split_data = load_split(config['split_file'])
    use_i2 = config.get('use_innovation_i2', False)
    train_loader = get_dataloader(
        config['data_root'], config['split_file'], split='train',
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 2),
        use_skeleton=use_i2
    )
    val_loader = get_dataloader(
        config['data_root'], config['split_file'], split='val',
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 2),
        use_skeleton=use_i2
    )

    # Compute pos_weight
    pos_weight = None
    if config.get('pos_weight') == 'auto':
        pos_weight = compute_pos_weight(config['data_root'], split_data['train'])
        print(f"[Train] pos_weight (auto): {pos_weight:.2f}")

    # Model
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = build_loss(config, pos_weight=pos_weight)
    optimizer  = get_optimizer(model, config)
    scheduler  = get_scheduler(optimizer, config)
    scaler     = GradScaler(enabled=(device.type == 'cuda'))

    # Logging
    writer = SummaryWriter(log_dir=run_dir)
    csv_path = os.path.join(run_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_dice,val_cldice,lr\n')

    # Training loop
    best_metric = -1.0
    patience_counter = 0
    early_stop_metric = config.get('early_stop_metric', 'val_dice')
    patience = config.get('early_stop_patience', 20)
    threshold = config.get('threshold', 0.5)
    epochs = config.get('epochs', 100)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion,
                                      optimizer, scaler, device, config)
        val_loss, val_agg = validate(model, val_loader, criterion,
                                      device, config, threshold=threshold)

        val_dice   = val_agg.get('dice', {}).get('mean', 0.0)
        val_cldice = val_agg.get('cldice', {}).get('mean', 0.0)
        lr = optimizer.param_groups[0]['lr']

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_dice)
        else:
            scheduler.step()

        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Dice/val',   val_dice,   epoch)
        writer.add_scalar('clDice/val', val_cldice, epoch)
        writer.add_scalar('LR',         lr,         epoch)

        with open(csv_path, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},'
                    f'{val_dice:.6f},{val_cldice:.6f},{lr:.8f}\n')

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_dice={val_dice:.4f} | val_cldice={val_cldice:.4f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        # Checkpoint
        metric_val = val_cldice if 'cldice' in early_stop_metric else val_dice
        if metric_val > best_metric:
            best_metric = metric_val
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'config': config,
            }, os.path.join(ckpt_dir, 'best.pth'))
            print(f"  → Best checkpoint saved (val_{early_stop_metric}={best_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    # Save last checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, os.path.join(ckpt_dir, 'last.pth'))

    writer.close()
    print(f"[Train] Done. Best val_{early_stop_metric}={best_metric:.4f}")
    return best_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args.config)


if __name__ == '__main__':
    main()
