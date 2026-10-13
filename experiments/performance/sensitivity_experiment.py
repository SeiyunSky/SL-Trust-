"""
参数敏感性实验：
  F1: Burn-in warmup 扫描 — warmup_rounds ∈ [20, 40, 60, 80], contra vs sltrust
  F2: Oscillating period 扫描 — period_attack ∈ [5, 10, 20], contra vs sltrust
  F4: S/L 超参网格扫描 — S∈{2,3,5,10}, L∈{5,10,20,40}，Burn-in 攻击下 SLTrust 精度热力图

展示随攻击参数变化，SLTrust 相比 CONTRA 的稳定性优势；以及 S/L 选择的合理区间。
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
from utils import get_result_dir
from experiment_io import load_cached_result, save_result

plt.rcParams['font.family'] = 'DejaVu Sans'

SENS_METHODS = ['contra', 'sltrust']
METHOD_LABELS = {
    'contra':  'CONTRA (EMA only)',
    'sltrust': 'SLTrust (dual-window, ours)',
}
METHOD_COLORS = {
    'contra':  '#FF9800',
    'sltrust': '#E53935',
}

BURNIN_WARMUPS = [20, 40, 60, 80]
OSC_PERIODS = [5, 10, 20]

# S/L 超参网格扫描配置
SL_S_VALUES = [2, 3, 5, 10]
SL_L_VALUES = [5, 10, 20, 40]


def _cache_path(save_dir: Path, dataset: str, method: str, attack: str,
                param_name: str, param_val, seed: int = None) -> Path:
    if seed is not None:
        d = save_dir / dataset / 'sensitivity' / f'{param_name}_{param_val}' / f'seed_{seed}'
    else:
        d = save_dir / dataset / 'sensitivity' / f'{param_name}_{param_val}'
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{method}_{attack}.json"


def run_single_combo(config_path=None):
    """子进程入口：跑单个 (method, attack, param)，结果存 JSON。"""
    method = os.environ.get('MVP_METHOD', 'sltrust')
    attack_name = os.environ.get('MVP_ATTACK', 'burn_in')
    seed_str = os.environ.get('MVP_SEED', '')
    warmup_str = os.environ.get('MVP_WARMUP', '')
    period_str = os.environ.get('MVP_PERIOD', '')

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'sensitivity.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # seed 覆写
    if seed_str:
        seed = int(seed_str)
        cfg['experiment']['seed'] = seed
        multi_seed = True
    else:
        seed = cfg['experiment']['seed']
        multi_seed = False

    # 攻击参数覆写
    if warmup_str:
        warmup = int(warmup_str)
        cfg['attack']['burn_in']['warmup_rounds'] = warmup
        param_name = 'burnin_warmup'
        param_val = warmup
    elif period_str:
        period = int(period_str)
        cfg['attack']['oscillating']['period_attack'] = period
        param_name = 'osc_period'
        param_val = period
    else:
        # 默认用 config 里的值
        if attack_name == 'burn_in':
            param_name = 'burnin_warmup'
            param_val = cfg['attack']['burn_in']['warmup_rounds']
        else:
            param_name = 'osc_period'
            param_val = cfg['attack']['oscillating']['period_attack']

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    out = _cache_path(save_dir, dataset, method, attack_name,
                      param_name, param_val, seed if multi_seed else None)
    extra = {'parameter': param_name, 'value': param_val}
    cached = load_cached_result(out, cfg, 'sensitivity', method, attack_name, extra)
    if cached is not None:
        print(f"[sensitivity | {method} | {attack_name} | {param_name}={param_val} | seed={seed}] SKIP complete cached result -> {out}")
        return cached

    print(f"[sensitivity | {method} | {attack_name} | {param_name}={param_val} | seed={seed}] Starting...")
    trainer = FederatedTrainer(cfg)
    history = trainer.train(method=method, attack_name=attack_name)
    save_result(out, history, cfg, 'sensitivity', method, attack_name, extra)
    print(f"[sensitivity | {method} | {attack_name} | {param_name}={param_val}] Done -> {out}")
    return history


def run_sensitivity(config_path=None):
    """加载缓存出图。"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'sensitivity.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    save_dir = Path(__file__).parent.parent.parent / 'results'

    print("=" * 72)
    print(f"SLTrust-DFL  Sensitivity  |  {dataset.upper()}")
    print("=" * 72)

    # F1: Burn-in warmup 扫描
    burnin_results = {m: {} for m in SENS_METHODS}
    for warmup in BURNIN_WARMUPS:
        for method in SENS_METHODS:
            # 尝试多 seed 目录
            base = save_dir / dataset / 'sensitivity' / f'burnin_warmup_{warmup}'
            seed_dirs = sorted(base.glob('seed_*')) if base.exists() else []
            accs = []
            for sd in seed_dirs:
                p = sd / f"{method}_burn_in.json"
                seed_cfg = json.loads(json.dumps(cfg))
                seed_cfg['experiment']['seed'] = int(sd.name.removeprefix('seed_'))
                seed_cfg['attack']['burn_in']['warmup_rounds'] = warmup
                h = load_cached_result(
                    p, seed_cfg, 'sensitivity', method, 'burn_in',
                    {'parameter': 'burnin_warmup', 'value': warmup})
                if h and h['test_acc']:
                    accs.append(h['test_acc'][-1])
            # 也看无 seed 子目录的
            p = base / f"{method}_burn_in.json"
            base_cfg = json.loads(json.dumps(cfg))
            base_cfg['attack']['burn_in']['warmup_rounds'] = warmup
            h = load_cached_result(
                p, base_cfg, 'sensitivity', method, 'burn_in',
                {'parameter': 'burnin_warmup', 'value': warmup})
            if h and h['test_acc']:
                accs.append(h['test_acc'][-1])
            if accs:
                burnin_results[method][warmup] = {
                    'mean': float(np.mean(accs)),
                    'std': float(np.std(accs)) if len(accs) > 1 else 0.0,
                }

    # F2: Oscillating period 扫描
    osc_results = {m: {} for m in SENS_METHODS}
    for period in OSC_PERIODS:
        for method in SENS_METHODS:
            base = save_dir / dataset / 'sensitivity' / f'osc_period_{period}'
            seed_dirs = sorted(base.glob('seed_*')) if base.exists() else []
            accs = []
            for sd in seed_dirs:
                p = sd / f"{method}_oscillating.json"
                seed_cfg = json.loads(json.dumps(cfg))
                seed_cfg['experiment']['seed'] = int(sd.name.removeprefix('seed_'))
                seed_cfg['attack']['oscillating']['period_attack'] = period
                h = load_cached_result(
                    p, seed_cfg, 'sensitivity', method, 'oscillating',
                    {'parameter': 'osc_period', 'value': period})
                if h and h['test_acc']:
                    accs.append(h['test_acc'][-1])
            p = base / f"{method}_oscillating.json"
            base_cfg = json.loads(json.dumps(cfg))
            base_cfg['attack']['oscillating']['period_attack'] = period
            h = load_cached_result(
                p, base_cfg, 'sensitivity', method, 'oscillating',
                {'parameter': 'osc_period', 'value': period})
            if h and h['test_acc']:
                accs.append(h['test_acc'][-1])
            if accs:
                osc_results[method][period] = {
                    'mean': float(np.mean(accs)),
                    'std': float(np.std(accs)) if len(accs) > 1 else 0.0,
                }

    _plot_sensitivity(burnin_results, osc_results, save_dir, dataset, cfg)
    _plot_sl_heatmap(save_dir, dataset, cfg)
    return burnin_results, osc_results


def _plot_sensitivity(burnin_results, osc_results, save_dir, dataset, cfg):
    d = get_result_dir(save_dir, dataset, 'sensitivity')
    byz = cfg['federated']['byzantine_ratio']
    alpha_noniid = cfg['data'].get('non_iid_alpha', 0.5)

    # 图1: Burn-in warmup vs accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in SENS_METHODS:
        xs, ys, errs = [], [], []
        for warmup in BURNIN_WARMUPS:
            info = burnin_results[method].get(warmup)
            if info:
                xs.append(warmup)
                ys.append(info['mean'])
                errs.append(info['std'])
        if xs:
            ax.errorbar(xs, ys, yerr=errs,
                        label=METHOD_LABELS[method],
                        color=METHOD_COLORS[method],
                        linewidth=2.5 if method == 'sltrust' else 1.8,
                        linestyle='-' if method == 'sltrust' else '--',
                        marker='o', markersize=8, capsize=4)
    if not any(burnin_results[method] for method in SENS_METHODS):
        plt.close()
        print("  sensitivity/burnin_warmup_sweep.png skipped (no verified data)")
    else:
        ax.set_xlabel('Burn-in Warmup Rounds', fontsize=12)
        ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
        ax.set_title(
            f'Sensitivity: Burn-in Warmup  |  {dataset.upper()}  |  '
            f'Byzantine={byz:.0%}  Non-IID α={alpha_noniid}',
            fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(BURNIN_WARMUPS)
        plt.tight_layout()
        plt.savefig(d / 'burnin_warmup_sweep.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  sensitivity/burnin_warmup_sweep.png saved")

    # 图2: Oscillating period vs accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in SENS_METHODS:
        xs, ys, errs = [], [], []
        for period in OSC_PERIODS:
            info = osc_results[method].get(period)
            if info:
                xs.append(period)
                ys.append(info['mean'])
                errs.append(info['std'])
        if xs:
            ax.errorbar(xs, ys, yerr=errs,
                        label=METHOD_LABELS[method],
                        color=METHOD_COLORS[method],
                        linewidth=2.5 if method == 'sltrust' else 1.8,
                        linestyle='-' if method == 'sltrust' else '--',
                        marker='s', markersize=8, capsize=4)
    if not any(osc_results[method] for method in SENS_METHODS):
        plt.close()
        print("  sensitivity/osc_period_sweep.png skipped (no verified data)")
    else:
        ax.set_xlabel('Oscillating Attack Period (rounds)', fontsize=12)
        ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
        ax.set_title(
            f'Sensitivity: Oscillating Period  |  {dataset.upper()}  |  '
            f'Byzantine={byz:.0%}  Non-IID α={alpha_noniid}',
            fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(OSC_PERIODS)
        plt.tight_layout()
        plt.savefig(d / 'osc_period_sweep.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  sensitivity/osc_period_sweep.png saved")

    # 终端汇总
    print("\n" + "=" * 60)
    print("Burn-in Warmup Sweep:")
    print(f"  {'Warmup':<10} {'CONTRA':>12} {'SLTrust':>12}")
    for w in BURNIN_WARMUPS:
        c = burnin_results['contra'].get(w, {}).get('mean')
        s = burnin_results['sltrust'].get(w, {}).get('mean')
        c_text = f'{c:.1f}%' if c is not None else '--'
        s_text = f'{s:.1f}%' if s is not None else '--'
        print(f"  {w:<10} {c_text:>12}  {s_text:>12}")
    print("\nOscillating Period Sweep:")
    print(f"  {'Period':<10} {'CONTRA':>12} {'SLTrust':>12}")
    for p in OSC_PERIODS:
        c = osc_results['contra'].get(p, {}).get('mean')
        s = osc_results['sltrust'].get(p, {}).get('mean')
        c_text = f'{c:.1f}%' if c is not None else '--'
        s_text = f'{s:.1f}%' if s is not None else '--'
        print(f"  {p:<10} {c_text:>12}  {s_text:>12}")
    print("=" * 60)


def run_sl_grid_single(config_path=None):
    """子进程入口：跑单个 (S, L) 组合，固定 burn_in 攻击"""
    method = 'sltrust'
    s_str = os.environ.get('MVP_S', '3')
    l_str = os.environ.get('MVP_L', '10')
    seed_str = os.environ.get('MVP_SEED', '')

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'sensitivity.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    S, L = int(s_str), int(l_str)
    cfg['sltrust']['S'] = S
    cfg['sltrust']['L'] = L

    if seed_str:
        seed = int(seed_str)
        cfg['experiment']['seed'] = seed
        multi_seed = True
    else:
        seed = cfg['experiment']['seed']
        multi_seed = False

    if S >= L:
        raise ValueError(f"SLTrust requires S < L, got S={S}, L={L}")

    dataset = cfg['data'].get('dataset', 'mnist').lower()
    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    d = save_dir / dataset / 'sensitivity' / f'sl_S{S}_L{L}'
    if multi_seed:
        d = d / f'seed_{seed}'
    out = d / 'sltrust_burn_in.json'
    extra = {'S': S, 'L': L}
    cached = load_cached_result(out, cfg, 'sensitivity_sl_grid', method, 'burn_in', extra)
    if cached is not None:
        print(f"[sl_grid | S={S} L={L} | seed={seed}] SKIP complete cached result -> {out}")
        return cached

    print(f"[sl_grid | S={S} L={L} | seed={seed}] Starting...")
    trainer = FederatedTrainer(cfg)
    history = trainer.train(method=method, attack_name='burn_in')
    save_result(out, history, cfg, 'sensitivity_sl_grid', method, 'burn_in', extra)
    print(f"[sl_grid | S={S} L={L} | seed={seed}] Done -> {out}")
    return history


def _plot_sl_heatmap(save_dir: Path, dataset: str, cfg: dict):
    """S/L 超参网格热力图：行=S值，列=L值，单元格=SLTrust在burn_in攻击下的最终精度"""
    d = get_result_dir(save_dir, dataset, 'sensitivity')
    byz = cfg['federated']['byzantine_ratio']
    alpha_noniid = cfg['data'].get('non_iid_alpha', 0.5)

    matrix = np.full((len(SL_S_VALUES), len(SL_L_VALUES)), np.nan)
    for i, S in enumerate(SL_S_VALUES):
        for j, L in enumerate(SL_L_VALUES):
            if S >= L:   # S必须小于L才有意义
                continue
            base = save_dir / dataset / 'sensitivity' / f'sl_S{S}_L{L}'
            accs = []
            # 多 seed
            for sd in sorted(base.glob('seed_*')):
                p = sd / 'sltrust_burn_in.json'
                seed_cfg = json.loads(json.dumps(cfg))
                seed_cfg['experiment']['seed'] = int(sd.name.removeprefix('seed_'))
                seed_cfg['sltrust']['S'] = S
                seed_cfg['sltrust']['L'] = L
                h = load_cached_result(
                    p, seed_cfg, 'sensitivity_sl_grid', 'sltrust', 'burn_in',
                    {'S': S, 'L': L})
                if h and h['test_acc']:
                    accs.append(h['test_acc'][-1])
            # 单 seed
            p = base / 'sltrust_burn_in.json'
            base_cfg = json.loads(json.dumps(cfg))
            base_cfg['sltrust']['S'] = S
            base_cfg['sltrust']['L'] = L
            h = load_cached_result(
                p, base_cfg, 'sensitivity_sl_grid', 'sltrust', 'burn_in',
                {'S': S, 'L': L})
            if h and h['test_acc']:
                accs.append(h['test_acc'][-1])
            if accs:
                matrix[i, j] = float(np.mean(accs))

    if np.all(np.isnan(matrix)):
        print("  sensitivity/sl_heatmap.png skipped (no S/L grid data)")
        return

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap='RdYlGn', aspect='auto',
                   vmin=max(10, vmin - 5), vmax=min(100, vmax + 2))
    plt.colorbar(im, ax=ax, label='Final Test Accuracy (%)')

    ax.set_xticks(range(len(SL_L_VALUES)))
    ax.set_xticklabels([f'L={l}' for l in SL_L_VALUES], fontsize=11)
    ax.set_yticks(range(len(SL_S_VALUES)))
    ax.set_yticklabels([f'S={s}' for s in SL_S_VALUES], fontsize=11)

    for i in range(len(SL_S_VALUES)):
        for j in range(len(SL_L_VALUES)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='black' if 30 < v < 85 else 'white')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=9, color='gray')

    ax.set_title(
        f'S/L Hyperparameter Grid  |  SLTrust  |  Burn-in Attack\n'
        f'{dataset.upper()}  Byzantine={byz:.0%}  Non-IID α={alpha_noniid}',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(d / 'sl_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  sensitivity/sl_heatmap.png saved")


if __name__ == '__main__':
    run_sensitivity()
