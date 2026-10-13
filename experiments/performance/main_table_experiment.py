"""
主表实验：8方法 × 10攻击
并行由 TUI 的 run_parallel 在进程层调度，本文件只负责单个 (method, attack) 组合。
通过环境变量 MVP_METHOD / MVP_ATTACK / MVP_SEED 接收参数。

目录结构：
  results/{dataset}_seed{seed}/
    main_table/{method}_{attack}.json
    main_table/heatmap.png
    main_table/summary_bar.png
    main_table/summary.json
    by_attack/{attack}/all_methods.png
    by_method/{method}/all_attacks.png
"""
import sys
import os
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trainer import FederatedTrainer
from utils import run_dir_name
from experiment_io import load_cached_result, save_result

plt.rcParams['font.family'] = 'DejaVu Sans'

MAIN_METHODS = ['fedavg', 'sama', 'balance', 'krum', 'trimmed_mean', 'contra', 'contra_hard', 'sltrust']
METHOD_LABELS = {
    'fedavg':       'FedAvg (baseline)',
    'sama':         'SAMA',
    'balance':      'BALANCE',
    'krum':         'Krum',
    'trimmed_mean': 'Trimmed Mean',
    'contra':       'CONTRA (soft EMA)',
    'contra_hard':  'CONTRA-Hard (hard cutoff)',
    'sltrust':      'SLTrust (ours)',
}
METHOD_COLORS = {
    'fedavg':       '#9E9E9E',
    'sama':         '#2196F3',
    'balance':      '#4CAF50',
    'krum':         '#9C27B0',
    'trimmed_mean': '#00BCD4',
    'contra':       '#FF9800',
    'contra_hard':  '#FF5722',
    'sltrust':      '#E53935',
}
METHOD_STYLES = {m: ('-' if m == 'sltrust' else '--') for m in MAIN_METHODS}

# ALIE 已从论文计划删除。Slow Poison 对 norm-aligned 方法与 Sign Flip
# 等价，但对不执行幅度对齐的基线仍可能不同，因此保留在完整主表中。
MAIN_ATTACKS = ['none', 'sign_flip', 'gaussian', 'burn_in', 'oscillating',
                'slow_poison', 'gradual_flip', 'backdoor']
ATTACK_LABELS = {
    'none':         'No Attack',
    'sign_flip':    'Sign Flip',
    'gaussian':     'Gaussian Noise',
    'alie':         'ALIE',
    'burn_in':      'Burn-in / Reputation Farming',
    'oscillating':  'Oscillating',
    'slow_poison':  'Slow Poison',
    'gradual_flip': 'Gradual Flip',
    'burn_in_slow': 'BurnIn+SlowPoison',
    'backdoor':     'Backdoor',
}
ATTACK_COLORS = {
    'none':         '#9E9E9E',
    'sign_flip':    '#E53935',
    'gaussian':     '#FF9800',
    'alie':         '#F06292',
    'burn_in':      '#2196F3',
    'oscillating':  '#4CAF50',
    'slow_poison':  '#00BCD4',
    'gradual_flip': '#9C27B0',
    'burn_in_slow': '#795548',
    'backdoor':     '#FF5722',
}


def _run_dir(save_dir: Path, dataset: str, seed: int) -> Path:
    return save_dir / run_dir_name(dataset, seed)


def run_single_combo(config_path=None):
    """子进程入口：跑单个 (method, attack)，结果存 JSON。"""
    method = os.environ.get('MVP_METHOD', 'sltrust')
    attack_name = os.environ.get('MVP_ATTACK', 'sign_flip')
    seed_str = os.environ.get('MVP_SEED', '')

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'main_table.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    seed = int(seed_str) if seed_str else cfg['experiment']['seed']
    cfg['experiment']['seed'] = seed

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    save_dir = Path(__file__).parent.parent.parent / 'results'
    out = _run_dir(save_dir, dataset, seed) / 'main_table' / f"{method}_{attack_name}.json"
    cached = load_cached_result(out, cfg, 'main_table', method, attack_name)
    if cached is not None:
        print(f"[{method} | {attack_name} | seed={seed}] SKIP complete cached result -> {out}")
        return cached

    print(f"[{method} | {attack_name} | seed={seed}] Starting...")
    trainer = FederatedTrainer(cfg)
    history = trainer.train(method=method, attack_name=attack_name)
    save_result(out, history, cfg, 'main_table', method, attack_name)
    print(f"[{method} | {attack_name} | seed={seed}] Done -> {out}")
    return history


def run_main_table(config_path=None):
    """出图入口：读取当前 seed 目录的全部 JSON，出完整图集。"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'main_table.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    seed = int(os.environ.get('MVP_SEED', cfg['experiment']['seed']))
    cfg['experiment']['seed'] = seed
    save_dir = Path(__file__).parent.parent.parent / 'results'
    run_root = _run_dir(save_dir, dataset, seed)

    num_rounds   = cfg['federated']['num_rounds']
    log_interval = cfg['logging']['log_interval']
    t_vals = np.arange(log_interval, num_rounds + 1, log_interval)

    print("=" * 72)
    print(f"SLTrust-DFL  Main Table  |  {dataset.upper()}  seed={seed}")
    print("=" * 72)

    # 加载当前 seed 目录的所有 JSON；串行入口补跑缺失组合。
    results = {a: {} for a in MAIN_ATTACKS}
    plot_only = os.environ.get('MVP_PLOT_ONLY', '0') == '1'
    for attack in MAIN_ATTACKS:
        for method in MAIN_METHODS:
            p = run_root / 'main_table' / f"{method}_{attack}.json"
            cached = load_cached_result(p, cfg, 'main_table', method, attack)
            if cached is not None:
                results[attack][method] = cached
            elif not plot_only:
                print(f"[{method} | {attack} | seed={seed}] Starting...")
                history = FederatedTrainer(cfg).train(method=method, attack_name=attack)
                save_result(p, history, cfg, 'main_table', method, attack)
                results[attack][method] = history

    _plot_by_attack(results, cfg, run_root, t_vals)
    _plot_heatmap(results, cfg, run_root)
    if os.environ.get('MVP_PLOT_DETAIL', 'compact') == 'full':
        _plot_by_method(results, cfg, run_root, t_vals)
        _plot_summary_bar(results, cfg, run_root)
    _save_summary(results, run_root)
    _print_summary(results)
    return results


def _plot_by_attack(results, cfg, run_root, t_vals):
    warmup = cfg.get('attack', {}).get('burn_in', {}).get('warmup_rounds', 40)
    byz = cfg['federated']['byzantine_ratio']
    alpha_noniid = cfg['data'].get('non_iid_alpha', 0.5)
    for attack in MAIN_ATTACKS:
        available = [m for m in MAIN_METHODS if results.get(attack, {}).get(m)]
        if not available:
            print(f"  by_attack/{attack}/all_methods.png skipped (no data)")
            continue
        d = run_root / 'by_attack' / attack
        d.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in MAIN_METHODS:
            hist = results.get(attack, {}).get(method)
            if not hist:
                continue
            acc = hist['test_acc']
            ax.plot(t_vals[:len(acc)], acc,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    linewidth=2.5 if method == 'sltrust' else 1.8,
                    linestyle=METHOD_STYLES[method],
                    alpha=0.9)
        if attack == 'burn_in':
            ax.axvline(x=warmup, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
            ymin, ymax = ax.get_ylim()
            ax.text(warmup + 1, ymin + (ymax - ymin) * 0.05, 'attack start', fontsize=9, alpha=0.7)
        ax.set_title(f"{ATTACK_LABELS[attack]}  |  Byzantine={byz:.0%}  Non-IID α={alpha_noniid}",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Test Accuracy (%)', fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(d / 'all_methods.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  by_attack/{attack}/all_methods.png saved")


def _plot_by_method(results, cfg, run_root, t_vals):
    byz = cfg['federated']['byzantine_ratio']
    alpha_noniid = cfg['data'].get('non_iid_alpha', 0.5)
    for method in MAIN_METHODS:
        available = [a for a in MAIN_ATTACKS if results.get(a, {}).get(method)]
        if not available:
            print(f"  by_method/{method}/all_attacks.png skipped (no data)")
            continue
        d = run_root / 'by_method' / method
        d.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        for attack in MAIN_ATTACKS:
            hist = results.get(attack, {}).get(method)
            if not hist:
                continue
            acc = hist['test_acc']
            ax.plot(t_vals[:len(acc)], acc,
                    label=ATTACK_LABELS[attack],
                    color=ATTACK_COLORS.get(attack, '#607D8B'),
                    linewidth=1.8, alpha=0.9)
        ax.set_title(f"{METHOD_LABELS[method]}  |  Byzantine={byz:.0%}  Non-IID α={alpha_noniid}",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Test Accuracy (%)', fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(d / 'all_attacks.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  by_method/{method}/all_attacks.png saved")


def _plot_heatmap(results, cfg, run_root):
    d = run_root / 'main_table'
    d.mkdir(parents=True, exist_ok=True)
    byz = cfg['federated']['byzantine_ratio']
    alpha_noniid = cfg['data'].get('non_iid_alpha', 0.5)
    matrix = np.full((len(MAIN_METHODS), len(MAIN_ATTACKS)), np.nan)
    for j, attack in enumerate(MAIN_ATTACKS):
        for i, method in enumerate(MAIN_METHODS):
            hist = results.get(attack, {}).get(method)
            if hist and hist.get('test_acc'):
                matrix[i, j] = hist['test_acc'][-1]
    finite = matrix[np.isfinite(matrix)]
    vmin = max(10, np.floor(finite.min() / 10) * 10) if finite.size else 10
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad(color='#E0E0E0')
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect='auto', vmin=vmin, vmax=100)
    plt.colorbar(im, ax=ax, label='Test Accuracy (%)').ax.tick_params(labelsize=10)
    ax.set_xticks(range(len(MAIN_ATTACKS)))
    ax.set_xticklabels([ATTACK_LABELS[a] for a in MAIN_ATTACKS], rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(MAIN_METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in MAIN_METHODS], fontsize=10)
    for i in range(len(MAIN_METHODS)):
        for j in range(len(MAIN_ATTACKS)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='#616161')
                continue
            tc = 'black' if vmin + (100 - vmin) * 0.25 < v < vmin + (100 - vmin) * 0.85 else 'white'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
    ax.set_title(f'Final Test Accuracy (%)  |  Byzantine={byz:.0%}  Non-IID α={alpha_noniid}',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(d / 'heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  main_table/heatmap.png saved")


def _plot_summary_bar(results, cfg, run_root):
    d = run_root / 'main_table'
    d.mkdir(parents=True, exist_ok=True)
    byz = cfg['federated']['byzantine_ratio']
    alpha_noniid = cfg['data'].get('non_iid_alpha', 0.5)
    x = np.arange(len(MAIN_ATTACKS))
    width = 0.8 / len(MAIN_METHODS)
    fig, ax = plt.subplots(figsize=(20, 7))
    for i, method in enumerate(MAIN_METHODS):
        vals = []
        for attack in MAIN_ATTACKS:
            hist = results.get(attack, {}).get(method)
            vals.append(hist['test_acc'][-1] if hist and hist.get('test_acc') else np.nan)
        ax.bar(x + (i - len(MAIN_METHODS) / 2 + 0.5) * width, vals, width,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS[a] for a in MAIN_ATTACKS], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title(f'Final Accuracy Summary  |  Byzantine={byz:.0%}  Non-IID α={alpha_noniid}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(d / 'summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  main_table/summary_bar.png saved")


def _save_summary(results, run_root):
    summary = {}
    for attack in MAIN_ATTACKS:
        summary[attack] = {}
        for method in MAIN_METHODS:
            hist = results.get(attack, {}).get(method)
            if hist and hist.get('test_acc'):
                summary[attack][method] = round(hist['test_acc'][-1], 4)
    p = run_root / 'main_table' / 'summary.json'
    with open(p, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  main_table/summary.json saved")


def _print_summary(results):
    print("\n" + "=" * 110)
    print(f"{'Method':<22}" + "".join(f" {a[:9]:>11}" for a in MAIN_ATTACKS))
    print("-" * 110)
    for method in MAIN_METHODS:
        row = f"  {METHOD_LABELS[method]:<20}"
        for attack in MAIN_ATTACKS:
            acc = results.get(attack, {}).get(method, {}).get('test_acc', [])
            row += f"  {acc[-1]:>8.2f}%" if acc else f"  {'--':>9}"
        print(row)
    print("=" * 110)


if __name__ == '__main__':
    run_main_table()
