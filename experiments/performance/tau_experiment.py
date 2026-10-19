"""
τ 演化实验：CONTRA vs SLTrust × 正式攻击矩阵
每轮记录对拜占庭邻居的平均τ（log_interval=1），每个攻击单独出一张τ演化曲线图。
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

TAU_METHODS = ['contra', 'sltrust']
METHOD_LABELS = {
    'contra':  'CONTRA (EMA only)',
    'sltrust': 'SLTrust (dual-window, ours)',
}
METHOD_COLORS = {
    'contra':  '#FF9800',
    'sltrust': '#E53935',
}

TAU_ATTACKS = [
    'gaussian', 'broken_node',
    'burn_in', 'oscillating', 'gradual_gaussian',
]
ATTACK_LABELS = {
    'gaussian':               'Gaussian Noise',
    'broken_node':            'Broken Node',
    'burn_in':                'Burn-in Gaussian',
    'oscillating':            'Oscillating Gaussian',
    'gradual_gaussian':       'Gradual Gaussian',
}


def _run_dir(save_dir: Path, dataset: str, seed: int) -> Path:
    return save_dir / run_dir_name(dataset, seed)


def run_single_combo(config_path=None):
    """子进程入口：跑单个 (method, attack)，结果存 JSON。"""
    method = os.environ.get('MVP_METHOD', 'sltrust')
    attack_name = os.environ.get('MVP_ATTACK', 'burn_in')
    seed_str = os.environ.get('MVP_SEED', '')

    if config_path is None:
        config_path = 'tau_mnist'

    cfg = load_experiment_config(config_path)

    seed = int(seed_str) if seed_str else cfg['experiment']['seed']
    cfg['experiment']['seed'] = seed

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    save_dir = Path(__file__).parent.parent.parent / 'results'
    out = _run_dir(save_dir, dataset, seed) / 'tau_evolution' / f"{method}_{attack_name}.json"
    cached = load_cached_result(out, cfg, 'tau_evolution', method, attack_name)
    if cached is not None:
        print(f"[tau | {method} | {attack_name} | seed={seed}] SKIP complete cached result -> {out}")
        return cached

    print(f"[tau | {method} | {attack_name} | seed={seed}] Starting...")
    trainer = FederatedTrainer(cfg)
    history = trainer.train(method=method, attack_name=attack_name)
    save_result(out, history, cfg, 'tau_evolution', method, attack_name)
    print(f"[tau | {method} | {attack_name} | seed={seed}] Done -> {out}")
    return history


def run_tau_experiment(config_path=None):
    """全量入口：加载缓存或串行回退，出图。"""
    if config_path is None:
        config_path = 'tau_mnist'

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
    print(f"SLTrust-DFL  τ Evolution  |  {dataset.upper()}  seed={seed}")
    print("=" * 72)

    results = {a: {} for a in TAU_ATTACKS}
    plot_only = os.environ.get('MVP_PLOT_ONLY', '0') == '1'
    for attack in TAU_ATTACKS:
        for method in TAU_METHODS:
            p = run_root / 'tau_evolution' / f"{method}_{attack}.json"
            cached = load_cached_result(p, cfg, 'tau_evolution', method, attack)
            if cached is not None:
                results[attack][method] = cached
            elif not plot_only:
                print(f"[tau | {method} | {attack} | seed={seed}] Starting...")
                history = FederatedTrainer(cfg).train(method=method, attack_name=attack)
                save_result(p, history, cfg, 'tau_evolution', method, attack)
                results[attack][method] = history

    _plot_tau_all(results, cfg, run_root, t_vals, dataset)
    _plot_tau_overview(results, cfg, run_root, t_vals, dataset)
    return results


def _plot_tau_all(results, cfg, run_root, t_vals, dataset):
    """每个攻击出一张τ演化曲线图，存 tau_evolution/{attack}.png"""
    import math
    warmup = cfg.get('attack', {}).get('burn_in', {}).get('warmup_rounds', 40)
    osc_period_attack = cfg.get('attack', {}).get('oscillating', {}).get('period_attack', 10)
    osc_period_rest   = cfg.get('attack', {}).get('oscillating', {}).get('period_rest', 15)

    for attack in TAU_ATTACKS:
        d = run_root / 'tau_evolution'
        d.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        has_data = False
        for method in TAU_METHODS:
            hist = results.get(attack, {}).get(method, {})
            tau_curve = hist.get('tau_byz', [])
            if not tau_curve:
                continue

            y = [v if not math.isnan(v) else None for v in tau_curve]
            xs = t_vals[:len(y)]

            # 分段绘制，跳过 None（无拜占庭邻居时为 nan）
            label_used = False
            seg_x, seg_y = [], []
            for xi, yi in zip(xs, y):
                if yi is not None:
                    seg_x.append(xi)
                    seg_y.append(yi)
                else:
                    if seg_x:
                        ax.plot(seg_x, seg_y,
                                color=METHOD_COLORS[method],
                                linewidth=2 if method == 'sltrust' else 1.5,
                                linestyle='-' if method == 'sltrust' else '--',
                                label=METHOD_LABELS[method] if not label_used else '_nolegend_')
                        label_used = True
                    seg_x, seg_y = [], []
            if seg_x:
                ax.plot(seg_x, seg_y,
                        color=METHOD_COLORS[method],
                        linewidth=2 if method == 'sltrust' else 1.5,
                        linestyle='-' if method == 'sltrust' else '--',
                        label=METHOD_LABELS[method] if not label_used else '_nolegend_')
            has_data = True

        if not has_data:
            plt.close()
            print(f"  tau_evolution/{attack}.png skipped (no tau data)")
            continue

        # 攻击边界标注
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        if attack == 'burn_in':
            ax.axvline(x=warmup, color='black', linestyle=':', linewidth=1.2, alpha=0.7)
            ax.text(warmup + 0.5, ymin + yrange * 0.92, 'attack start',
                    fontsize=8, color='black', alpha=0.8)
        elif attack == 'oscillating':
            period = osc_period_attack + osc_period_rest
            t = 0
            first = True
            while t < t_vals[-1]:
                ax.axvspan(t, t + osc_period_attack, alpha=0.08, color='red')
                if first:
                    ax.axvspan(t, t + osc_period_attack, alpha=0.08, color='red',
                               label='attack phase')
                    first = False
                t += period

        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Avg τ (Byzantine neighbors)', fontsize=11)
        ax.set_title(f'τ Evolution — {ATTACK_LABELS[attack]}  |  {dataset.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(d / f'{attack}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  tau_evolution/{attack}.png saved")


def _plot_tau_overview(results, cfg, run_root, t_vals, dataset):
    """全攻击 τ overview：2行×4列子图网格，一图展示所有攻击下 CONTRA vs SLTrust τ 走势"""
    import math as _math
    warmup = cfg.get('attack', {}).get('burn_in', {}).get('warmup_rounds', 40)
    attacks = [a for a in TAU_ATTACKS if any(
        results.get(a, {}).get(m, {}).get('tau_byz') for m in TAU_METHODS
    )]
    if not attacks:
        print("  tau_overview.png skipped (no data)")
        return

    n = len(attacks)
    ncols = 4
    nrows = _math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), squeeze=False)

    for idx, attack in enumerate(attacks):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        has_data = False
        for method in TAU_METHODS:
            hist = results.get(attack, {}).get(method, {})
            tau_curve = hist.get('tau_byz', [])
            if not tau_curve:
                continue
            y = [v if not _math.isnan(v) else None for v in tau_curve]
            xs = t_vals[:len(y)]
            # 简单绘制（跳过 None 段）
            seg_x, seg_y = [], []
            label_used = False
            for xi, yi in zip(xs, y):
                if yi is not None:
                    seg_x.append(xi)
                    seg_y.append(yi)
                else:
                    if seg_x:
                        ax.plot(seg_x, seg_y, color=METHOD_COLORS[method],
                                linewidth=1.8 if method == 'sltrust' else 1.2,
                                linestyle='-' if method == 'sltrust' else '--',
                                label=METHOD_LABELS[method] if not label_used else '_nolegend_')
                        label_used = True
                    seg_x, seg_y = [], []
            if seg_x:
                ax.plot(seg_x, seg_y, color=METHOD_COLORS[method],
                        linewidth=1.8 if method == 'sltrust' else 1.2,
                        linestyle='-' if method == 'sltrust' else '--',
                        label=METHOD_LABELS[method] if not label_used else '_nolegend_')
            has_data = True

        if attack == 'burn_in':
            ax.axvline(x=warmup, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
        ax.set_title(ATTACK_LABELS.get(attack, attack), fontsize=9, fontweight='bold')
        ax.set_xlabel('Round', fontsize=8)
        ax.set_ylabel('Avg τ (Byz)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if has_data and idx == 0:
            ax.legend(fontsize=7)

    # 隐藏多余子图
    for idx in range(len(attacks), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f'τ Evolution Overview — {dataset.upper()}  (CONTRA vs SLTrust)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    d = run_root / 'tau_evolution'
    d.mkdir(parents=True, exist_ok=True)
    plt.savefig(d / 'tau_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  tau_evolution/tau_overview.png saved")

if __name__ == '__main__':
    run_tau_experiment()
