"""
Final evaluation script. Runs on the TEST SET ONLY with saved checkpoints.

Usage:
    python -m src.evaluate --config configs/baseline.yaml --split test
    python -m src.evaluate --config configs/hybrid_i2.yaml --split test
    python -m src.evaluate --all   # evaluate all experiments and generate comparison table

Outputs:
    results/tables/{experiment_name}_metrics.csv
    results/tables/comparison_table.csv
    results/tables/comparison_table.tex
    results/tables/ablation_table.csv
    Paired Wilcoxon test results (M1 vs M2+I2)
"""
import os, sys, json, yaml, argparse
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from scipy.stats import wilcoxon

from src.datasets import get_dataloader, load_split
from src.metrics import per_image_metrics, aggregate_metrics
from src.postprocess import postprocess
from src.models.hybrid_unet import build_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(config, device):
    ckpt_dir = config['checkpoint_dir']
    ckpt_path = os.path.join(ckpt_dir, 'best.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Eval] Loaded checkpoint: {ckpt_path}")
    return model


@torch.no_grad()
def evaluate_model(model, loader, device, config):
    """
    Run model on all images in loader. Returns per-image metric list
    and raw data for statistical tests.
    """
    threshold = config.get('threshold', 0.5)
    min_comp  = config.get('min_component_size', 50)
    apply_cl  = config.get('apply_closing', False)
    use_i2    = config.get('use_innovation_i2', False)

    per_image_results = []
    raw_probs  = []
    raw_preds  = []
    raw_gts    = []
    image_ids  = []

    for batch in loader:
        images     = batch['image'].to(device)
        masks      = batch['mask']
        pad_params = batch['pad_params']
        ids        = batch['image_id']

        with autocast(enabled=(device.type == 'cuda')):
            if use_i2:
                logits, _ = model(images)
            else:
                logits = model(images)

        probs   = torch.sigmoid(logits).cpu().numpy()
        masks_np = masks.numpy()

        for i in range(images.shape[0]):
            top  = pad_params[0][i].item() if torch.is_tensor(pad_params[0][i]) else pad_params[0][i]
            bot  = pad_params[1][i].item() if torch.is_tensor(pad_params[1][i]) else pad_params[1][i]
            left = pad_params[2][i].item() if torch.is_tensor(pad_params[2][i]) else pad_params[2][i]
            right= pad_params[3][i].item() if torch.is_tensor(pad_params[3][i]) else pad_params[3][i]

            H, W = probs.shape[2], probs.shape[3]
            prob_crop = probs[i, 0, top:H-bot, left:W-right]
            mask_crop = masks_np[i, 0, top:H-bot, left:W-right]

            pred_crop = (prob_crop > threshold).astype(np.uint8)
            pred_pp   = postprocess(pred_crop, min_component_size=min_comp,
                                     apply_closing=apply_cl)

            m = per_image_metrics(pred_pp, mask_crop.astype(np.uint8),
                                   prob_map=prob_crop)
            per_image_results.append(m)
            raw_probs.append(prob_crop)
            raw_preds.append(pred_pp)
            raw_gts.append(mask_crop.astype(np.uint8))
            image_ids.append(ids[i].item() if torch.is_tensor(ids[i]) else ids[i])

    return per_image_results, raw_probs, raw_preds, raw_gts, image_ids


def wilcoxon_test(metrics_a, metrics_b, metric_name):
    """
    Paired Wilcoxon signed-rank test for metric_name.
    Returns (statistic, p_value, effect_size).
    """
    vals_a = [m[metric_name] for m in metrics_a]
    vals_b = [m[metric_name] for m in metrics_b]
    diffs  = [a - b for a, b in zip(vals_a, vals_b)]
    if all(d == 0 for d in diffs):
        return 0.0, 1.0, 0.0
    stat, p = wilcoxon(vals_a, vals_b, alternative='two-sided')
    # Rank-biserial correlation as effect size
    n = len(diffs)
    effect_size = stat / (n * (n + 1) / 2)
    return float(stat), float(p), float(effect_size)


def format_table(agg_dict, experiment_name):
    """Convert aggregate dict to a DataFrame row."""
    metrics_to_report = [
        'dice', 'iou', 'precision', 'recall', 'specificity',
        'f1', 'boundary_f1', 'hd95', 'asd',
        'cldice', 'topo_precision', 'topo_recall',
        'n_components_pred', 'component_diff',
        'accuracy', 'mean_class_acc',
        'auc_roc', 'auc_pr',
        'thin_recall', 'thick_recall',
    ]
    row = {'experiment': experiment_name}
    for m in metrics_to_report:
        if m in agg_dict:
            mean = agg_dict[m]['mean']
            std  = agg_dict[m]['std']
            row[m] = f"{mean:.4f} ± {std:.4f}"
            row[f'{m}_mean'] = mean
            row[f'{m}_std']  = std
    return row


def evaluate_all(config_paths, split='test', output_dir='results/tables'):
    """Evaluate all models and generate comparison table."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_rows = []
    all_per_image = {}

    for cfg_path in config_paths:
        config = load_config(cfg_path)
        exp_name = config.get('experiment_name', os.path.basename(cfg_path))
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")

        model  = load_checkpoint(config, device)
        loader = get_dataloader(
            config['data_root'], config['split_file'],
            split=split, batch_size=1, num_workers=2
        )

        per_image, probs, preds, gts, ids = evaluate_model(model, loader, device, config)
        agg = aggregate_metrics(per_image)
        all_per_image[exp_name] = per_image

        # Save per-image CSV
        df_per = pd.DataFrame(per_image)
        df_per.insert(0, 'image_id', ids)
        df_per.to_csv(os.path.join(output_dir, f'{exp_name}_per_image.csv'), index=False)

        row = format_table(agg, exp_name)
        all_rows.append(row)

        # Print summary
        print(f"  Dice:   {agg.get('dice', {}).get('mean', 0):.4f} ± {agg.get('dice', {}).get('std', 0):.4f}")
        print(f"  clDice: {agg.get('cldice', {}).get('mean', 0):.4f} ± {agg.get('cldice', {}).get('std', 0):.4f}")
        print(f"  HD95:   {agg.get('hd95', {}).get('mean', 0):.4f}")

    # Comparison table
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)

    # LaTeX table
    metric_cols = ['dice', 'iou', 'precision', 'recall', 'specificity',
                   'cldice', 'hd95', 'asd', 'auc_roc', 'auc_pr']
    display_cols = ['experiment'] + [m for m in metric_cols if m in df.columns]
    latex = df[display_cols].to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, 'comparison_table.tex'), 'w') as f:
        f.write(latex)

    # Wilcoxon test (M1 vs M2+I2 if both present)
    exp_names = list(all_per_image.keys())
    if len(exp_names) >= 2:
        a_name = exp_names[0]   # assume M1 is first
        b_name = exp_names[-1]  # assume M2+I2 is last
        print(f"\n{'='*60}")
        print(f"Wilcoxon test: {a_name} vs {b_name}")
        wil_results = []
        for metric_name in ['dice', 'cldice']:
            try:
                stat, p, eff = wilcoxon_test(
                    all_per_image[a_name], all_per_image[b_name], metric_name
                )
                wil_results.append({
                    'metric': metric_name,
                    'statistic': stat, 'p_value': p, 'effect_size': eff
                })
                print(f"  {metric_name}: stat={stat:.4f}, p={p:.4f}, effect_size={eff:.4f}")
            except Exception as e:
                print(f"  {metric_name}: test failed ({e})")
        pd.DataFrame(wil_results).to_csv(
            os.path.join(output_dir, 'wilcoxon_results.csv'), index=False)

    print(f"\n[Eval] Tables saved to {output_dir}/")
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Single config path')
    parser.add_argument('--all',    action='store_true', help='Evaluate all experiments')
    parser.add_argument('--split',  default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', default='results/tables')
    args = parser.parse_args()

    if args.all or args.config is None:
        config_paths = [
            'configs/baseline.yaml',
            'configs/hybrid.yaml',
            'configs/hybrid_i2.yaml',
        ]
        evaluate_all(config_paths, split=args.split, output_dir=args.output_dir)
    else:
        config = load_config(args.config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = load_checkpoint(config, device)
        loader = get_dataloader(
            config['data_root'], config['split_file'],
            split=args.split, batch_size=1, num_workers=2
        )
        per_image, _, _, _, ids = evaluate_model(model, loader, device, config)
        agg = aggregate_metrics(per_image)
        for k, v in agg.items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == '__main__':
    main()
