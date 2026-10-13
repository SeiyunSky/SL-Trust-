"""
MVP 验证实验：4方法 × 2攻击，确认双窗口有效性
并行由 TUI 的 run_parallel 在进程层调度，本文件只负责单个 (method, attack) 组合
通过环境变量 MVP_METHOD / MVP_ATTACK 接收参数（与 SAMA 实验的 ATTACK_TYPE 惯例一致）
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
from experiment_io import load_cached_result, save_result
from utils import run_dir_name
plt.rcParams['font.family'] = 'DejaVu Sans'


MVP_METHODS = ['fedavg', 'sama', 'contra', 'sltrust']
METHOD_LABELS = {
    'fedavg':  'FedAvg (baseline)',
    'sama':    'SAMA (no memory)',
    'contra':  'CONTRA (EMA only)',
    'sltrust': 'SLTrust (dual-window, ours)',
}
METHOD_COLORS = {
    'fedavg':  '#999999',
    'sama':    '#2196F3',
    'contra':  '#FF9800',
    'sltrust': '#E53935',
}
MVP_ATTACKS = ['sign_flip', 'burn_in', 'oscillating']
ATTACK_LABELS = {
    'sign_flip':   'Sign Flip (persistent)',
    'burn_in':     'Burn-in / Reputation Farming',
    'oscillating': 'Oscillating (attack/rest cycle)',
}


def run_single_combo(config_path=None):
    """
    单个 (method, attack) 组合入口，由子进程调用。
    从环境变量 MVP_METHOD / MVP_ATTACK 读取参数，结果存 JSON。
    """
    method = os.environ.get('MVP_METHOD', 'sltrust')
    attack_name = os.environ.get('MVP_ATTACK', 'sign_flip')
    seed_str = os.environ.get('MVP_SEED', '')

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'mvp.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    seed = int(seed_str) if seed_str else cfg['experiment']['seed']
    cfg['experiment']['seed'] = seed
    save_dir = Path(__file__).parent.parent.parent / 'results'
    result_dir = save_dir / run_dir_name(dataset, seed) / 'mvp'
    result_dir.mkdir(parents=True, exist_ok=True)
    out = result_dir / f"{method}_{attack_name}.json"
    cached = load_cached_result(out, cfg, 'mvp', method, attack_name)
    if cached is not None:
        print(f"[{method} | {attack_name}] SKIP complete cached result -> {out.name}")
        return cached

    print(f"[{method} | {attack_name}] Starting...")
    trainer = FederatedTrainer(cfg)
    history = trainer.train(method=method, attack_name=attack_name)
    save_result(out, history, cfg, 'mvp', method, attack_name)
    print(f"[{method} | {attack_name}] Done -> {out.name}")
    return history


def run_mvp_experiment(config_path=None):
    """
    全量入口：串行跑所有组合，最后统一出图。
    TUI 并行模式下不走这里——每个子进程各自调用 run_single_combo。
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'mvp.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)

    num_rounds = cfg['federated']['num_rounds']
    log_interval = cfg['logging']['log_interval']
    t_vals = np.arange(log_interval, num_rounds + 1, log_interval)

    print("=" * 72)
    print("SLTrust-DFL  MVP Validation Experiment")
    print(f"Methods: {MVP_METHODS}")
    print(f"Attacks: {MVP_ATTACKS}")
    print("=" * 72)

    # 尝试从已有 JSON 文件加载（TUI 并行跑完后合并出图用）
    dataset = cfg['data'].get('dataset', 'mnist').lower()
    seed = cfg['experiment']['seed']
    result_dir = save_dir / run_dir_name(dataset, seed) / 'mvp'
    results = {a: {} for a in MVP_ATTACKS}
    all_cached = True
    for attack in MVP_ATTACKS:
        for method in MVP_METHODS:
            p = result_dir / f"{method}_{attack}.json"
            cached = load_cached_result(p, cfg, 'mvp', method, attack)
            if cached is not None:
                results[attack][method] = cached
            else:
                all_cached = False

    if not all_cached and os.environ.get('MVP_PLOT_ONLY', '0') != '1':
        # 串行回退：直接跑
        trainer = FederatedTrainer(cfg)
        for attack in MVP_ATTACKS:
            for method in MVP_METHODS:
                if method not in results[attack]:
                    history = trainer.train(method=method, attack_name=attack)
                    p = result_dir / f"{method}_{attack}.json"
                    save_result(p, history, cfg, 'mvp', method, attack)
                    results[attack][method] = history

    _plot_and_save(results, cfg, result_dir, t_vals)
    return results


def _plot_and_save(results, cfg, save_dir, t_vals):
    num_rounds = cfg['federated']['num_rounds']
    log_interval = cfg['logging']['log_interval']
    dataset = cfg['data'].get('dataset', 'mnist').upper()
    n_attacks = len(MVP_ATTACKS)

    # ── 主图：2行 × n_attacks列（精度曲线 + 柱状图）
    fig, axes = plt.subplots(2, n_attacks, figsize=(6 * n_attacks, 10))
    if n_attacks == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        f"SLTrust-DFL MVP  |  {dataset}  |  Non-IID α={cfg['data']['non_iid_alpha']}  "
        f"|  Byzantine={cfg['federated']['byzantine_ratio']:.0%}",
        fontsize=13, fontweight='bold'
    )

    for col, attack in enumerate(MVP_ATTACKS):
        ax_curve = axes[0, col]
        ax_bar   = axes[1, col]
        ax_curve.set_title(f"Attack: {ATTACK_LABELS[attack]}", fontsize=11)
        ax_bar.set_title(f"Final Accuracy — {ATTACK_LABELS[attack]}", fontsize=11)

        final_accs = []
        for method in MVP_METHODS:
            if method not in results.get(attack, {}):
                final_accs.append(np.nan)
                continue
            hist = results[attack][method]
            acc_curve = hist['test_acc']
            ax_curve.plot(t_vals[:len(acc_curve)], acc_curve,
                          label=METHOD_LABELS[method],
                          color=METHOD_COLORS[method],
                          linewidth=2 if method == 'sltrust' else 1.5,
                          linestyle='-' if method == 'sltrust' else '--',
                          alpha=0.9)
            final_accs.append(acc_curve[-1] if acc_curve else np.nan)

        ax_curve.set_xlabel('Round')
        ax_curve.set_ylabel('Test Accuracy (%)')
        ax_curve.legend(fontsize=9)
        ax_curve.grid(True, alpha=0.3)

        if attack == 'burn_in':
            warmup = cfg.get('attack', {}).get('burn_in', {}).get('warmup_rounds', 40)
            ax_curve.axvline(x=warmup, color='black', linestyle=':', linewidth=1, alpha=0.6)
            ax_curve.text(warmup + 1, ax_curve.get_ylim()[0] + 1, 'attack start',
                          fontsize=8, color='black', alpha=0.7)

        x = np.arange(len(MVP_METHODS))
        bars = ax_bar.bar(x, final_accs,
                          color=[METHOD_COLORS[m] for m in MVP_METHODS], alpha=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([METHOD_LABELS[m].split(' ')[0] for m in MVP_METHODS],
                               fontsize=9)
        ax_bar.set_ylabel('Test Accuracy (%)')
        ax_bar.grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, final_accs):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{acc:.1f}%', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_path = save_dir / f'mvp_validation_{dataset.lower()}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {fig_path}")

    # ── τ 演化曲线：burn_in 下 CONTRA vs SLTrust 对拜占庭邻居的平均 τ
    _plot_tau_evolution(results, cfg, save_dir, t_vals, dataset)

    # ── 终端汇总
    print("\n" + "=" * 72)
    header = f"{'Method':<30}" + "".join(f" {ATTACK_LABELS[a][:12]:>14}" for a in MVP_ATTACKS)
    print(header)
    print("-" * 72)
    for method in MVP_METHODS:
        row = f"  {METHOD_LABELS[method]:<28}"
        for attack in MVP_ATTACKS:
            acc_list = results.get(attack, {}).get(method, {}).get('test_acc', [])
            row += f"  {acc_list[-1]:>12.2f}%" if acc_list else f"  {'--':>13}"
        print(row)
    print("=" * 72)


def _plot_tau_evolution(results, cfg, save_dir, t_vals, dataset='MNIST'):
    """图2：burn_in 攻击下 CONTRA vs SLTrust 对拜占庭邻居的平均 τ 演化"""
    burn_in_results = results.get('burn_in', {})
    warmup = cfg.get('attack', {}).get('burn_in', {}).get('warmup_rounds', 40)

    tau_methods = ['contra', 'sltrust']
    tau_colors  = {'contra': '#FF9800', 'sltrust': '#E53935'}
    tau_labels  = {'contra': 'CONTRA (EMA only)', 'sltrust': 'SLTrust (dual-window, ours)'}

    # 检查是否有 tau_byz 数据
    has_tau = any(
        burn_in_results.get(m, {}).get('tau_byz')
        for m in tau_methods
    )
    if not has_tau:
        print("  [τ plot] No tau_byz data found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in tau_methods:
        hist = burn_in_results.get(method, {})
        tau_curve = hist.get('tau_byz', [])
        if not tau_curve:
            continue
        import math
        y = [v if not math.isnan(v) else None for v in tau_curve]
        xs = t_vals[:len(y)]
        label_used = False
        seg_x, seg_y = [], []
        for xi, yi in zip(xs, y):
            if yi is not None:
                seg_x.append(xi)
                seg_y.append(yi)
            else:
                if seg_x:
                    ax.plot(seg_x, seg_y, color=tau_colors[method],
                            linewidth=2 if method == 'sltrust' else 1.5,
                            linestyle='-' if method == 'sltrust' else '--',
                            label=tau_labels[method] if not label_used else '_nolegend_')
                    label_used = True
                seg_x, seg_y = [], []
        if seg_x:
            ax.plot(seg_x, seg_y, color=tau_colors[method],
                    linewidth=2 if method == 'sltrust' else 1.5,
                    linestyle='-' if method == 'sltrust' else '--',
                    label=tau_labels[method] if not label_used else '_nolegend_')

    ax.axvline(x=warmup, color='black', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(warmup + 1, ax.get_ylim()[1] * 0.95, 'attack start',
            fontsize=9, color='black', alpha=0.8)
    ax.set_xlabel('Round')
    ax.set_ylabel('Avg τ (Byzantine neighbors)')
    ax.set_title(f'Reputation Evolution under Burn-in Attack  |  {dataset}', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path = save_dir / f'tau_evolution_{dataset.lower()}.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"τ evolution plot saved: {fig_path}")


if __name__ == '__main__':
    run_mvp_experiment()
