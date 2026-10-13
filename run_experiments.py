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
                        help='单组合模式：指定攻击（sign_flip/burn_in）')
    args = parser.parse_args()

    sys.path.insert(0, PERF_PATH)

    if args.experiment.startswith('main_table'):
        module = importlib.import_module('main_table_experiment')
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

    cifar_main = Path(__file__).parent / 'configs' / 'main_table_cifar10.yaml'
    cifar_tau  = Path(__file__).parent / 'configs' / 'tau.yaml'   # tau 暂用 mnist config，cifar 版单独建
    cifar_abl  = Path(__file__).parent / 'configs' / 'ablation.yaml'

    def _set_env():
        if args.method: os.environ['MVP_METHOD'] = args.method
        if args.attack:  os.environ['MVP_ATTACK'] = args.attack

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
        module.run_mvp_experiment(config_path=Path(__file__).parent / 'configs' / 'mvp_cifar10.yaml')
    elif args.experiment == 'mvp_cifar10_single':
        _set_env(); module.run_single_combo(config_path=Path(__file__).parent / 'configs' / 'mvp_cifar10.yaml')
    elif args.experiment == 'mvp_cifar10_plot':
        _plot_only(lambda: module.run_mvp_experiment(config_path=Path(__file__).parent / 'configs' / 'mvp_cifar10.yaml'))

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

    elif args.experiment == 'tau_mnist':
        _plot_only(module.run_tau_experiment) if os.environ.get('MVP_PLOT_ONLY') == '1' else module.run_tau_experiment()
    elif args.experiment == 'tau_mnist_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'tau_cifar10':
        cfg_tau_cifar = Path(__file__).parent / 'configs' / 'tau_cifar10.yaml'
        if os.environ.get('MVP_PLOT_ONLY') == '1':
            _plot_only(lambda: module.run_tau_experiment(config_path=cfg_tau_cifar))
        else:
            module.run_tau_experiment(config_path=cfg_tau_cifar)
    elif args.experiment == 'tau_cifar10_single':
        _set_env(); module.run_single_combo(config_path=Path(__file__).parent / 'configs' / 'tau_cifar10.yaml')

    elif args.experiment == 'tau_decay_mnist':
        _read_results(module.run_tau_decay)
    elif args.experiment == 'tau_decay_mnist_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'tau_decay_cifar10':
        _read_results(lambda: module.run_tau_decay(config_path=Path(__file__).parent / 'configs' / 'tau_cifar10.yaml'))
    elif args.experiment == 'tau_decay_cifar10_single':
        _set_env(); module.run_single_combo(config_path=Path(__file__).parent / 'configs' / 'tau_cifar10.yaml')

    elif args.experiment == 'ablation_mnist':
        module.run_ablation()
    elif args.experiment == 'ablation_single':
        _set_env(); module.run_single_combo()
    elif args.experiment == 'ablation_plot':
        _plot_only(module.run_ablation)
    elif args.experiment == 'ablation_cifar10':
        cfg_abl_cifar = Path(__file__).parent / 'configs' / 'ablation_cifar10.yaml'
        module.run_ablation(config_path=cfg_abl_cifar)
    elif args.experiment == 'ablation_cifar10_single':
        _set_env(); module.run_single_combo(config_path=Path(__file__).parent / 'configs' / 'ablation_cifar10.yaml')
    elif args.experiment == 'ablation_cifar10_plot':
        _plot_only(lambda: module.run_ablation(config_path=Path(__file__).parent / 'configs' / 'ablation_cifar10.yaml'))

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
