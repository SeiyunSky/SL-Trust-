"""MNIST pressure tests for Non-IID, topology, and Byzantine ratio."""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiment_io import load_cached_result, save_result
from config_loader import load_experiment_config
from experiment_matrix import apply_stress_value, decode_value, stress_tasks
from utils import run_dir_name

ROOT = Path(__file__).parent.parent.parent


def run_single_combo(config_path):
    base_cfg = load_experiment_config(config_path)
    method = os.environ['MVP_METHOD']
    attack = os.environ['MVP_ATTACK']
    seed = int(os.environ.get('MVP_SEED', base_cfg['experiment']['seed']))
    value = decode_value(os.environ['MVP_STRESS_VALUE'])
    cfg, label = apply_stress_value(base_cfg, value)
    cfg['experiment']['seed'] = seed
    dimension = cfg['stress']['dimension']
    dataset = cfg['data']['dataset'].lower()
    out = (ROOT / 'results' / run_dir_name(dataset, seed) / 'stress' /
           dimension / label / f'{method}_{attack}.json')
    extra = {'dimension': dimension, 'value': value}
    cached = load_cached_result(out, cfg, f'stress_{dimension}', method, attack, extra)
    if cached is not None:
        print(f'[{dimension}={label} | {method} | {attack} | seed={seed}] SKIP -> {out}')
        return cached
    from trainer import FederatedTrainer
    history = FederatedTrainer(cfg).train(method, attack)
    save_result(out, history, cfg, f'stress_{dimension}', method, attack, extra)
    print(f'[{dimension}={label} | {method} | {attack} | seed={seed}] Done -> {out}')
    return history


def run_summary(config_path):
    """Create a cross-seed final-accuracy plot from verified stress results."""
    base_cfg = load_experiment_config(config_path)
    dimension = base_cfg['stress']['dimension']
    dataset = base_cfg['data']['dataset'].lower()
    stress = base_cfg['stress']
    values = stress['values']
    labels = [apply_stress_value(base_cfg, value)[1] for value in values]
    series = {(method, attack): [] for method in stress['methods']
              for attack in stress['attacks']}
    for value, label in zip(values, labels):
        for method in stress['methods']:
            for attack in stress['attacks']:
                finals = []
                for seed in stress['seeds']:
                    cfg, _ = apply_stress_value(base_cfg, value)
                    cfg['experiment']['seed'] = int(seed)
                    path = (ROOT / 'results' / run_dir_name(dataset, int(seed)) /
                            'stress' / dimension / label / f'{method}_{attack}.json')
                    history = load_cached_result(
                        path, cfg, f'stress_{dimension}', method, attack,
                        {'dimension': dimension, 'value': value})
                    if history is not None:
                        finals.append(float(history['test_acc'][-1]))
                series[(method, attack)].append(
                    (float(np.mean(finals)), float(np.std(finals)), len(finals))
                    if finals else (np.nan, np.nan, 0))

    out_dir = ROOT / 'results' / dataset / 'stress' / dimension
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(stress['attacks']),
                             figsize=(5 * len(stress['attacks']), 4), squeeze=False)
    x = np.arange(len(values))
    for column, attack in enumerate(stress['attacks']):
        ax = axes[0, column]
        for method in stress['methods']:
            points = series[(method, attack)]
            means = np.array([point[0] for point in points])
            stds = np.array([point[1] for point in points])
            ax.errorbar(x, means, yerr=stds, marker='o', capsize=3, label=method)
        ax.set_title(attack)
        ax.set_xticks(x, labels, rotation=20, ha='right')
        ax.set_ylabel('Final test accuracy (%)')
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle(f'{dataset.upper()} {dimension} stress test (mean ± std across seeds)')
    fig.tight_layout()
    out = out_dir / 'final_accuracy.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    return series


def print_manifest(config_path):
    cfg = load_experiment_config(config_path)
    tasks = stress_tasks(cfg)
    print(f"dimension={cfg['stress']['dimension']} unique_tasks={len(tasks)}")
    for task in tasks:
        print(task)
    return tasks
