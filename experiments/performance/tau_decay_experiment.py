"""
τ 衰减速率定量实验
在 Burn-in 攻击下，记录 CONTRA / CONTRA-Hard / SLTrust / SLTrust-M 四种方法
的对拜占庭邻居 τ 逐轮变化，统计"τ 首次降至阈值以下所需轮数"，输出柱状图。

对命题5.4 的直接实验支撑："SLTrust 的 τ 衰减速率严格快于 CONTRA"
"""
import sys
import os
import json
import numpy as np
import math
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

DECAY_METHODS = ['contra', 'contra_hard', 'sltrust', 'sltrust_m']
METHOD_LABELS = {
    'contra':      'CONTRA\n(soft EMA)',
    'contra_hard': 'CONTRA-Hard\n(hard cutoff)',
    'sltrust':     'SLTrust\n(ours)',
    'sltrust_m':   'SLTrust-M\n(median, ours)',
}
METHOD_COLORS = {
    'contra':      '#FF9800',
    'contra_hard': '#FF5722',
    'sltrust':     '#E53935',
    'sltrust_m':   '#B71C1C',
}

DECAY_THRESHOLD = 0.1


def _run_dir(save_dir: Path, dataset: str, seed: int) -> Path:
    return save_dir / run_dir_name(dataset, seed)


def _cache_path(save_dir: Path, dataset: str, method: str, seed: int) -> Path:
    d = _run_dir(save_dir, dataset, seed) / 'tau_decay'
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{method}_burn_in.json"


def _method_to_exp_key(method: str) -> str:
    """将 decay method 名映射到 trainer 里的方法名"""
    if method == 'sltrust_m':
        return 'sltrust_median'
    return method


def _make_sltrust_m_cfg(cfg: dict) -> dict:
    """在 config 里注入 sltrust_median 配置（复用 sltrust 超参，mode=median_bg）"""
    import copy
    c = copy.deepcopy(cfg)
    base = c.get('sltrust', {})
    c['sltrust_median'] = dict(base, mode='median_bg')
    return c


def run_single_combo(config_path=None):
    """子进程入口，由 tui 通过环境变量调度"""
    method = os.environ.get('MVP_METHOD', 'sltrust')
    seed_str = os.environ.get('MVP_SEED', '')

    if config_path is None:
        config_path = 'tau_mnist'

    cfg = load_experiment_config(config_path)

    # tau_decay 需要每轮记录，强制 log_interval=1
    cfg['logging']['log_interval'] = 1

    seed = int(seed_str) if seed_str else cfg['experiment']['seed']
    cfg['experiment']['seed'] = seed

    # sltrust_m 用 median_bg mode
    if method == 'sltrust_m':
        cfg = _make_sltrust_m_cfg(cfg)
        trainer_method = 'sltrust_median'
    else:
        trainer_method = method

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    save_dir = Path(__file__).parent.parent.parent / 'results'
    out = _cache_path(save_dir, dataset, method, seed)
    cached = load_cached_result(out, cfg, 'tau_decay', trainer_method, 'burn_in', {'label': method})
    if cached is not None:
        print(f"[tau_decay | {method} | burn_in | seed={seed}] SKIP complete cached result -> {out}")
        return cached

    # CONTRA/SLTrust 的 τ evolution 使用相同配置和逐轮日志，可直接复用。
    if method in {'contra', 'sltrust'}:
        tau_path = _run_dir(save_dir, dataset, seed) / 'tau_evolution' / f'{method}_burn_in.json'
        reused = load_cached_result(tau_path, cfg, 'tau_evolution', method, 'burn_in')
        if reused is not None:
            save_result(out, reused, cfg, 'tau_decay', trainer_method, 'burn_in', {'label': method})
            print(f"[tau_decay | {method} | burn_in | seed={seed}] REUSE tau evolution -> {out}")
            return reused

    print(f"[tau_decay | {method} | burn_in | seed={seed}] Starting...")
    trainer = FederatedTrainer(cfg)
    history = trainer.train(method=trainer_method, attack_name='burn_in')
    save_result(out, history, cfg, 'tau_decay', trainer_method, 'burn_in', {'label': method})
    print(f"[tau_decay | {method} | burn_in | seed={seed}] Done -> {out}")
    return history


def _first_below(tau_curve: list, threshold: float, warmup: int) -> int:
    """返回 warmup 之后 τ 首次低于 threshold 的轮次（绝对轮次），找不到返回 None"""
    for i, v in enumerate(tau_curve):
        if i < warmup:
            continue
        if not math.isnan(v) and v < threshold:
            return i  # log_interval=1，index = 轮次
    return None


def run_tau_decay(config_path=None):
    """全量入口：加载多 seed 结果，统计衰减速率，出柱状图"""
    if config_path is None:
        config_path = 'tau_mnist'

    cfg = load_experiment_config(config_path)

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    warmup = cfg.get('attack', {}).get('burn_in', {}).get('warmup_rounds', 40)
    save_dir = Path(__file__).parent.parent.parent / 'results'

    print("=" * 72)
    print(f"SLTrust-DFL  τ Decay Rate  |  {dataset.upper()}  |  threshold={DECAY_THRESHOLD}")
    print("=" * 72)

    # 收集所有 seed 的衰减轮次；仅接受指纹匹配且完整的结果。
    cfg['logging']['log_interval'] = 1
    decay_rounds: dict[str, list] = {m: [] for m in DECAY_METHODS}
    run_dirs = list(save_dir.glob(f"{dataset}_seed*/tau_decay"))
    run_dirs += list(save_dir.glob(f"{dataset}_*_seed*/tau_decay"))  # legacy dated layout
    for run_dir in sorted(set(run_dirs)):
        seed_text = run_dir.parent.name.rsplit('_seed', 1)[-1]
        if not seed_text.isdigit():
            continue
        for method in DECAY_METHODS:
            p = run_dir / f"{method}_burn_in.json"
            method_cfg = json.loads(json.dumps(cfg))
            method_cfg['experiment']['seed'] = int(seed_text)
            trainer_method = _method_to_exp_key(method)
            if method == 'sltrust_m':
                method_cfg = _make_sltrust_m_cfg(method_cfg)
            hist = load_cached_result(
                p, method_cfg, 'tau_decay', trainer_method, 'burn_in',
                {'label': method})
            if hist is None:
                continue
            tau_curve = hist.get('tau_byz', [])
            r = _first_below(tau_curve, DECAY_THRESHOLD, warmup)
            # 未在观测窗口内越过阈值属于右删失，而不是缺失运行。
            decay_rounds[method].append(
                r if r is not None else cfg['federated']['num_rounds'] + 1)

    _plot_decay_bar(decay_rounds, save_dir, dataset, warmup)
    _print_decay_summary(decay_rounds, warmup)
    return decay_rounds


def _plot_decay_bar(decay_rounds: dict, save_dir: Path, dataset: str, warmup: int):
    d = save_dir / dataset / 'tau_decay'
    d.mkdir(parents=True, exist_ok=True)

    means, stds, labels, colors = [], [], [], []
    for method in DECAY_METHODS:
        vals = decay_rounds[method]
        if not vals:
            means.append(np.nan)
            stds.append(0)
        else:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        labels.append(METHOD_LABELS[method])
        colors.append(METHOD_COLORS[method])

    x = np.arange(len(DECAY_METHODS))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, width=0.55, error_kw={'elinewidth': 1.5})

    for bar, mean, std in zip(bars, means, stds):
        if np.isfinite(mean):
            label = f'{mean:.1f}' if std == 0 else f'{mean:.1f}±{std:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.5,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.5, 'N/A',
                    ha='center', va='bottom', fontsize=10, color='#616161')

    # 标注 warmup 结束线（attack start）
    ax.axhline(y=warmup, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.text(len(DECAY_METHODS) - 0.45, warmup + 0.3, 'attack start (warmup end)',
            fontsize=8, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(f'Rounds to τ < {DECAY_THRESHOLD}  (lower = faster; non-crossing capped)', fontsize=11)
    ax.set_title(
        f'τ Decay Speed after Burn-in Attack  |  {dataset.upper()}\n'
        f'(Non-crossing runs are capped at num_rounds + 1)',
        fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(d / 'decay_rounds_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  tau_decay/decay_rounds_bar.png saved")

    # 同时保存原始数据
    out = d / 'decay_rounds.json'
    with open(out, 'w') as f:
        json.dump({m: decay_rounds[m] for m in DECAY_METHODS}, f, indent=2)
    print(f"  tau_decay/decay_rounds.json saved")


def _print_decay_summary(decay_rounds: dict, warmup: int):
    print("\n" + "=" * 60)
    print(f"τ Decay Summary  (attack starts at round {warmup})")
    print(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'N':>4}")
    print("-" * 60)
    for method in DECAY_METHODS:
        vals = decay_rounds[method]
        if vals:
            print(f"  {METHOD_LABELS[method].replace(chr(10),' '):<18} "
                  f"{np.mean(vals):>8.1f} {np.std(vals) if len(vals)>1 else 0:>8.1f} {len(vals):>4}")
        else:
            print(f"  {METHOD_LABELS[method].replace(chr(10),' '):<18} {'--':>8} {'--':>8} {'0':>4}")
    print("=" * 60)


if __name__ == '__main__':
    run_tau_decay()
