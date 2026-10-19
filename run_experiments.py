"""
Main runner — simplified for refactor-v2
Supports single-job subprocess mode via --experiment and env vars.
"""
import argparse
import sys
import importlib
import io
import os
from pathlib import Path

os.environ.setdefault('PYTHONUTF8', '1')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True, write_through=True)
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True, write_through=True)
except (AttributeError, io.UnsupportedOperation):
    pass

sys.path.insert(0, str(Path(__file__).parent))

PERF_PATH = str(Path(__file__).parent / 'experiments' / 'performance')


def main():
    parser = argparse.ArgumentParser(description='SLTrust-DFL Experiments')
    parser.add_argument('--experiment', type=str, default='main_table_single',
                        choices=[
                            'main_table_single', 'main_table_plot',
                            'main_table_cifar10_single', 'main_table_cifar10_plot',
                            'ablation_single', 'ablation_plot',
                            'ablation_cifar10_single', 'ablation_cifar10_plot',
                            'tau_mnist', 'tau_mnist_single',
                            'tau_cifar10', 'tau_cifar10_single',
                            'sensitivity_single', 'sensitivity_plot',
                        ])
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--attack', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--period', type=int, default=None)
    args = parser.parse_args()

    sys.path.insert(0, PERF_PATH)

    if args.experiment.startswith('main_table'):
        module = importlib.import_module('main_table_experiment')
    elif args.experiment.startswith('tau'):
        module = importlib.import_module('tau_experiment')
    elif args.experiment.startswith('ablation'):
        module = importlib.import_module('ablation_experiment')
    elif args.experiment.startswith('sensitivity'):
        module = importlib.import_module('sensitivity_experiment')
    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)

    def _set_env():
        if args.method: os.environ['MVP_METHOD'] = args.method
        if args.attack: os.environ['MVP_ATTACK'] = args.attack
        if args.seed is not None: os.environ['MVP_SEED'] = str(args.seed)
        if args.warmup is not None: os.environ['MVP_WARMUP'] = str(args.warmup)
        if args.period is not None: os.environ['MVP_PERIOD'] = str(args.period)

    def _plot_only(callable_):
        prev_force = os.environ.get('MVP_FORCE')
        prev_plot = os.environ.get('MVP_PLOT_ONLY')
        os.environ['MVP_FORCE'] = '0'
        os.environ['MVP_PLOT_ONLY'] = '1'
        try:
            return callable_()
        finally:
            if prev_force is None: os.environ.pop('MVP_FORCE', None)
            else: os.environ['MVP_FORCE'] = prev_force
            if prev_plot is None: os.environ.pop('MVP_PLOT_ONLY', None)
            else: os.environ['MVP_PLOT_ONLY'] = prev_plot

    exp = args.experiment

    if exp == 'main_table_single':
        _set_env(); module.run_single_combo()
    elif exp == 'main_table_plot':
        _plot_only(module.run_main_table)
    elif exp == 'main_table_cifar10_single':
        _set_env(); module.run_single_combo(config_path='main_table_cifar10')
    elif exp == 'main_table_cifar10_plot':
        _plot_only(lambda: module.run_main_table(config_path='main_table_cifar10'))

    elif exp == 'ablation_single':
        _set_env(); module.run_single_combo()
    elif exp == 'ablation_plot':
        _plot_only(module.run_ablation)
    elif exp == 'ablation_cifar10_single':
        _set_env(); module.run_single_combo(config_path='ablation_cifar10')
    elif exp == 'ablation_cifar10_plot':
        _plot_only(lambda: module.run_ablation(config_path='ablation_cifar10'))

    elif exp == 'tau_mnist':
        _plot_only(module.run_tau_experiment) if os.environ.get('MVP_PLOT_ONLY') == '1' else module.run_tau_experiment()
    elif exp == 'tau_mnist_single':
        _set_env(); module.run_single_combo()
    elif exp == 'tau_cifar10':
        if os.environ.get('MVP_PLOT_ONLY') == '1':
            _plot_only(lambda: module.run_tau_experiment(config_path='tau_cifar10'))
        else:
            module.run_tau_experiment(config_path='tau_cifar10')
    elif exp == 'tau_cifar10_single':
        _set_env(); module.run_single_combo(config_path='tau_cifar10')

    elif exp == 'sensitivity_single':
        _set_env(); module.run_single_combo()
    elif exp == 'sensitivity_plot':
        prev_force = os.environ.get('MVP_FORCE')
        os.environ['MVP_FORCE'] = '0'
        try:
            module.run_sensitivity()
        finally:
            if prev_force is None: os.environ.pop('MVP_FORCE', None)
            else: os.environ['MVP_FORCE'] = prev_force


if __name__ == '__main__':
    main()
