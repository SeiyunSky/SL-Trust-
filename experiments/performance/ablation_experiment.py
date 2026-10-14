"""
消融实验：SLTrust 内部4变体 × 2攻击（burn_in / oscillating）
独立实验，不依附主表，证明双窗口 ratio 惩罚是关键组件。
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trainer import FederatedTrainer
from utils import run_dir_name
from experiment_io import load_cached_result, save_result
from config_loader import load_experiment_config

plt.rcParams['font.family'] = 'DejaVu Sans'

ABLATION_METHODS = ['sltrust', 'sltrust_no_ratio', 'sltrust_short_only', 'sltrust_long_only']
METHOD_LABELS = {
    'sltrust':            'SLTrust (full, dual-window)',
    'sltrust_no_ratio':   'SLTrust-noRatio (EMA only)',
    'sltrust_short_only': 'SLTrust-shortOnly (no long memory)',
    'sltrust_long_only':  'SLTrust-longOnly (no short window)',
}
METHOD_COLORS = {
    'sltrust':            '#E53935',
    'sltrust_no_ratio':   '#FF9800',
    'sltrust_short_only': '#2196F3',
    'sltrust_long_only':  '#4CAF50',
}

ABLATION_ATTACKS = ['burn_in', 'oscillating']
ATTACK_LABELS = {
    'burn_in':     'Burn-in / Reputation Farming',
    'oscillating': 'Oscillating',
}


def _run_dir(save_dir: Path, dataset: str, seed: int) -> Path:
    return save_dir / run_dir_name(dataset, seed)


def run_single_combo(config_path=None):
    """子进程入口：跑单个 (method, attack)，结果存 JSON。"""
    method = os.environ.get('MVP_METHOD', 'sltrust')
    attack_name = os.environ.get('MVP_ATTACK', 'burn_in')
    seed_str = os.environ.get('MVP_SEED', '')

    if config_path is None:
        config_path = 'ablation_mnist'

    cfg = load_experiment_config(config_path)

    seed = int(seed_str) if seed_str else cfg['experiment']['seed']
    cfg['experiment']['seed'] = seed

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    save_dir = Path(__file__).parent.parent.parent / 'results'
    out = _run_dir(save_dir, dataset, seed) / 'ablation' / f"{method}_{attack_name}.json"
    cached = load_cached_result(out, cfg, 'ablation', method, attack_name)
    if cached is not None:
        print(f"[ablation | {method} | {attack_name} | seed={seed}] SKIP complete cached result -> {out}")
        return cached

    print(f"[ablation | {method} | {attack_name} | seed={seed}] Starting...")
    trainer = FederatedTrainer(cfg)
    history = trainer.train(method=method, attack_name=attack_name)
    save_result(out, history, cfg, 'ablation', method, attack_name)
    print(f"[ablation | {method} | {attack_name} | seed={seed}] Done -> {out}")
    return history


def run_ablation(config_path=None):
    """全量入口：加载缓存或串行回退，出图。"""
    if config_path is None:
        config_path = 'ablation_mnist'

    cfg = load_experiment_config(config_path)

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    seed = int(os.environ.get('MVP_SEED', cfg['experiment']['seed']))
    cfg['experiment']['seed'] = seed
    save_dir = Path(__file__).parent.parent.parent / 'results'
    run_root = _run_dir(save_dir, dataset, seed)

    num_rounds   = cfg['federated']['num_rounds']
    log_interval = cfg['logging']['log_interval']
    t_vals = np.arange(log_interval, num_rounds + 1, log_interval)

    print("=" * 72)
    print(f"SLTrust-DFL  Ablation  |  {dataset.upper()}  seed={seed}")
    print("=" * 72)

    results = {a: {} for a in ABLATION_ATTACKS}
    plot_only = os.environ.get('MVP_PLOT_ONLY', '0') == '1'
    for attack in ABLATION_ATTACKS:
        for method in ABLATION_METHODS:
            p = run_root / 'ablation' / f"{method}_{attack}.json"
            cached = load_cached_result(p, cfg, 'ablation', method, attack)
            if cached is not None:
                results[attack][method] = cached
            elif not plot_only:
                print(f"[ablation | {method} | {attack} | seed={seed}] Starting...")
                history = FederatedTrainer(cfg).train(method=method, attack_name=attack)
                save_result(p, history, cfg, 'ablation', method, attack)
                results[attack][method] = history

    _plot_ablation(results, cfg, run_root, t_vals, dataset)
    return results


def _plot_ablation(results, cfg, run_root, t_vals, dataset):
    d = run_root / 'ablation'
    d.mkdir(parents=True, exist_ok=True)
    warmup = cfg.get('attack', {}).get('burn_in', {}).get('warmup_rounds', 40)

    # 图1/图2：每个攻击单独一张精度曲线
    for attack in ABLATION_ATTACKS:
        if not any(results.get(attack, {}).get(method) for method in ABLATION_METHODS):
            print(f"  ablation/{attack}.png skipped (no data)")
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in ABLATION_METHODS:
            hist = results.get(attack, {}).get(method)
            if not hist:
                continue
            acc = hist['test_acc']
            ax.plot(t_vals[:len(acc)], acc,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    linewidth=2.5 if method == 'sltrust' else 1.8,
                    linestyle='-' if method == 'sltrust' else '--',
                    alpha=0.9)
        if attack == 'burn_in':
            ax.axvline(x=warmup, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
            ymin, ymax = ax.get_ylim()
            ax.text(warmup + 0.5, ymin + (ymax - ymin) * 0.05,
                    'attack start', fontsize=9, alpha=0.7)
        ax.set_title(f'Ablation — {ATTACK_LABELS[attack]}  |  {dataset.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Test Accuracy (%)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(d / f'{attack}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ablation/{attack}.png saved")

    # 图3：2攻击 × 4变体 最终精度柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    n_attacks = len(ABLATION_ATTACKS)
    n_methods = len(ABLATION_METHODS)
    x = np.arange(n_attacks)
    width = 0.7 / n_methods
    for i, method in enumerate(ABLATION_METHODS):
        vals = []
        for attack in ABLATION_ATTACKS:
            hist = results.get(attack, {}).get(method)
            vals.append(hist['test_acc'][-1] if hist and hist['test_acc'] else np.nan)
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS[a] for a in ABLATION_ATTACKS], fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title(f'Ablation Final Accuracy  |  {dataset.upper()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(d / 'ablation_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ablation/ablation_bar.png saved")

    # 终端汇总
    print("\n" + "=" * 60)
    print(f"{'Method':<30} {'BurnIn':>10} {'Oscillating':>12}")
    print("-" * 60)
    for method in ABLATION_METHODS:
        bi  = results.get('burn_in', {}).get(method, {}).get('test_acc', [])
        osc = results.get('oscillating', {}).get(method, {}).get('test_acc', [])
        bi_text = f'{bi[-1]:.2f}%' if bi else '--'
        osc_text = f'{osc[-1]:.2f}%' if osc else '--'
        print(f"  {METHOD_LABELS[method]:<28} {bi_text:>9}  {osc_text:>11}")
    print("=" * 60)


if __name__ == '__main__':
    run_ablation()
