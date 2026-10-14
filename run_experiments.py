"""
Main runner — 与 SAMA 项目同接口
支持单组合子进程模式（--experiment mvp_single --method xxx --attack xxx）
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
    parser.add_argument('--experiment', type=str, default='mvp',
                        choices=[
                            'mvp', 'mvp_single', 'mvp_plot',
                            'mvp_cifar10', 'mvp_cifar10_single', 'mvp_cifar10_plot',
                            'main_table', 'main_table_single', 'main_table_plot',
                            'main_table_cifar10', 'main_table_cifar10_single', 'main_table_cifar10_plot',
                            'generalization_cifar100', 'generalization_cifar100_single', 'generalization_cifar100_plot',
                            'generalization_cifar100_manifest',
                            'stress_non_iid_single', 'stress_topology_single', 'stress_byzantine_ratio_single',
                            'stress_non_iid_manifest', 'stress_topology_manifest', 'stress_byzantine_ratio_manifest',
                            'stress_non_iid_plot', 'stress_topology_plot', 'stress_byzantine_ratio_plot',
                            'tau_mnist', 'tau_mnist_single', 'tau_cifar10', 'tau_cifar10_single',
                            'tau_decay_mnist', 'tau_decay_mnist_single',
                            'tau_decay_cifar10', 'tau_decay_cifar10_single',
                            'ablation_mnist', 'ablation_single', 'ablation_plot',
                            'ablation_cifar10', 'ablation_cifar10_single', 'ablation_cifar10_plot',
                            'sensitivity_single', 'sensitivity_plot', 'sensitivity_sl_single',
                        ])
    parser.add_argument('--method', type=str, default=None,
                        help='单组合模式：指定方法（fedavg/sama/contra/sltrust）')
    parser.add_argument('--attack', type=str, default=None,
                        help='单组合模式：指定攻击（例如 label_flipping/burn_in）')
    parser.add_argument('--seed', type=int, default=None,
                        help='覆盖单组合任务 seed')
    parser.add_argument('--warmup', type=int, default=None,
                        help='sensitivity single：覆盖 Burn-in warm-up rounds')
    parser.add_argument('--period', type=int, default=None,
                        help='sensitivity single：覆盖 oscillating attack period')
    parser.add_argument('--lambda-bd', type=float, default=None,
                        help='sensitivity single：覆盖 direction perturbation strength')
    parser.add_argument('--stress-value', type=str, default=None,
                        help='压力测试值的 JSON，例如 0.1 或 {"type":"ring"}')
    args = parser.parse_args()

    sys.path.insert(0, PERF_PATH)

    if args.experiment.startswith(('main_table', 'generalization_cifar100')):
        module = importlib.import_module('main_table_experiment')
    elif args.experiment.startswith('stress_'):
        module = importlib.import_module('stress_experiment')
    elif args.experiment.startswith('tau_decay'):
        module = importlib.import_module('tau_decay_experiment')
    elif args.experiment.startswith('tau'):
        module = importlib.import_module('tau_experiment')
    elif args.experiment.startswith('ablation'):
        module = importlib.import_module('ablation_experiment')
    elif args.experiment.startswith('sensitivity'):
        module = importlib.import_module('sensitivity_experiment')
    else:
        module = importlib.import_module('mvp_experiment')

    cifar_main = 'main_table_cifar10'
    cifar100_generalization = 'generalization_cifar100'

    def _set_env():
        if args.method: os.environ['MVP_METHOD'] = args.method
        if args.attack:  os.environ['MVP_ATTACK'] = args.attack
        if args.seed is not None: os.environ['MVP_SEED'] = str(args.seed)
        if args.warmup is not None: os.environ['MVP_WARMUP'] = str(args.warmup)
        if args.period is not None: os.environ['MVP_PERIOD'] = str(args.period)
        if args.lambda_bd is not None: os.environ['MVP_LAMBDA'] = str(args.lambda_bd)
        if args.stress_value is not None: os.environ['MVP_STRESS_VALUE'] = args.stress_value

    def _read_results(callable_, plot_only=False):
        previous_force = os.environ.get('MVP_FORCE')
        previous_plot = os.environ.get('MVP_PLOT_ONLY')
        os.environ['MVP_FORCE'] = '0'
        if plot_only:
            os.environ['MVP_PLOT_ONLY'] = '1'
        try:
            return callable_()
        finally:
            if previous_force is None:
                os.environ.pop('MVP_FORCE', None)
            else:
                os.environ['MVP_FORCE'] = previous_force
            if previous_plot is None:
                os.environ.pop('MVP_PLOT_ONLY', None)
            else:
                os.environ['MVP_PLOT_ONLY'] = previous_plot

    def _plot_only(callable_):
        return _read_results(callable_, plot_only=True)

    if args.experiment == 'mvp':
        module.run_mvp_experiment()
    elif args.experiment == 'mvp_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'mvp_plot':
        _plot_only(module.run_mvp_experiment)
    elif args.experiment == 'mvp_cifar10':
        module.run_mvp_experiment(config_path='mvp_cifar10')
    elif args.experiment == 'mvp_cifar10_single':
        _set_env(); module.run_single_combo(config_path='mvp_cifar10')
    elif args.experiment == 'mvp_cifar10_plot':
        _plot_only(lambda: module.run_mvp_experiment(config_path='mvp_cifar10'))

    elif args.experiment == 'main_table':
        module.run_main_table()
    elif args.experiment == 'main_table_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'main_table_plot':
        _plot_only(module.run_main_table)
    elif args.experiment == 'main_table_cifar10':
        module.run_main_table(config_path=cifar_main)
    elif args.experiment == 'main_table_cifar10_single':
        _set_env(); module.run_single_combo(config_path=cifar_main)
    elif args.experiment == 'main_table_cifar10_plot':
        _plot_only(lambda: module.run_main_table(config_path=cifar_main))
    elif args.experiment == 'generalization_cifar100':
        module.run_configured_seeds(config_path=cifar100_generalization)
    elif args.experiment == 'generalization_cifar100_single':
        _set_env(); module.run_single_combo(config_path=cifar100_generalization)
    elif args.experiment == 'generalization_cifar100_plot':
        _plot_only(lambda: module.run_main_table(config_path=cifar100_generalization))
    elif args.experiment == 'generalization_cifar100_manifest':
        module.print_manifest(config_path=cifar100_generalization)

    elif args.experiment.startswith('stress_'):
        dimension = (args.experiment.removeprefix('stress_')
                     .removesuffix('_single').removesuffix('_manifest').removesuffix('_plot'))
        config_path = f'stress_{dimension}_mnist'
        if args.experiment.endswith('_manifest'):
            module.print_manifest(config_path)
        elif args.experiment.endswith('_plot'):
            module.run_summary(config_path)
        else:
            _set_env()
            if not args.method or not args.attack or args.stress_value is None:
                parser.error('stress single requires --method, --attack, and --stress-value')
            module.run_single_combo(config_path)

    elif args.experiment == 'tau_mnist':
        _plot_only(module.run_tau_experiment) if os.environ.get('MVP_PLOT_ONLY') == '1' else module.run_tau_experiment()
    elif args.experiment == 'tau_mnist_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'tau_cifar10':
        if os.environ.get('MVP_PLOT_ONLY') == '1':
            _plot_only(lambda: module.run_tau_experiment(config_path='tau_cifar10'))
        else:
            module.run_tau_experiment(config_path='tau_cifar10')
    elif args.experiment == 'tau_cifar10_single':
        _set_env(); module.run_single_combo(config_path='tau_cifar10')

    elif args.experiment == 'tau_decay_mnist':
        _read_results(module.run_tau_decay)
    elif args.experiment == 'tau_decay_mnist_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'tau_decay_cifar10':
        _read_results(lambda: module.run_tau_decay(config_path='tau_cifar10'))
    elif args.experiment == 'tau_decay_cifar10_single':
        _set_env(); module.run_single_combo(config_path='tau_cifar10')

    elif args.experiment == 'ablation_mnist':
        module.run_ablation()
    elif args.experiment == 'ablation_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'ablation_plot':
        _plot_only(module.run_ablation)
    elif args.experiment == 'ablation_cifar10':
        module.run_ablation(config_path='ablation_cifar10')
    elif args.experiment == 'ablation_cifar10_single':
        _set_env(); module.run_single_combo(config_path='ablation_cifar10')
    elif args.experiment == 'ablation_cifar10_plot':
        _plot_only(lambda: module.run_ablation(config_path='ablation_cifar10'))

    elif args.experiment == 'sensitivity_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'sensitivity_sl_single':
        _set_env(); module.run_sl_grid_single()
    elif args.experiment == 'sensitivity_plot':
        _read_results(module.run_sensitivity)

    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)


if __name__ == '__main__':
    main()
