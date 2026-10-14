import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from config_loader import load_experiment_config
from experiment_matrix import (
    apply_stress_value, main_table_tasks, sensitivity_tasks, stress_tasks,
)


ROOT = Path(__file__).parent.parent


class ExperimentMatrixTest(unittest.TestCase):
    @staticmethod
    def _load(name):
        return load_experiment_config(name.removesuffix('.yaml'))

    def test_cifar100_matrix_has_exactly_84_unique_tasks(self):
        tasks = main_table_tasks(self._load('generalization_cifar100'))
        keys = {(t['seed'], t['method'], t['attack']) for t in tasks}
        self.assertEqual(84, len(tasks))
        self.assertEqual(len(tasks), len(keys))

    def test_configured_seed_runner_executes_every_yaml_seed(self):
        from experiments.performance import main_table_experiment

        cfg = self._load('generalization_cifar100')
        expected = [str(seed) for seed in cfg['matrix']['seeds']]
        seen = []
        with patch.object(main_table_experiment, 'run_main_table',
                          side_effect=lambda _: seen.append(os.environ['MVP_SEED'])):
            main_table_experiment.run_configured_seeds(
                'generalization_cifar100')
        self.assertEqual(expected, seen)

    def test_sensitivity_tasks_are_yaml_defined_and_unique(self):
        tasks = sensitivity_tasks(self._load('sensitivity_mnist'))
        keys = {(t['seed'], t['method'], t['attack'], t['parameter'], t['value'])
                for t in tasks}
        self.assertEqual(24, len(tasks))
        self.assertEqual(len(tasks), len(keys))
        self.assertEqual({0, 20, 40, 60}, {
            task['value'] for task in tasks if task['parameter'] == 'warmup'
        })
        self.assertEqual({0.05, 0.1, 0.2, 0.3, 0.5}, {
            task['value'] for task in tasks if task['parameter'] == 'lambda'
        })

    def test_stress_task_counts_are_yaml_defined_and_unique(self):
        for name in ('stress_non_iid_mnist', 'stress_topology_mnist',
                     'stress_byzantine_ratio_mnist'):
            cfg = self._load(name)
            stress = cfg['stress']
            expected = (len(stress['values']) * len(stress['methods']) *
                        len(stress['attacks']) * len(stress['seeds']))
            tasks = stress_tasks(cfg)
            keys = {
                (t['seed'], t['method'], t['attack'], repr(t['value']))
                for t in tasks
            }
            self.assertEqual(expected, len(tasks), name)
            self.assertEqual(len(tasks), len(keys), name)

    def test_stress_value_only_changes_selected_dimension(self):
        cfg = self._load('stress_non_iid_mnist')
        changed, label = apply_stress_value(cfg, 0.1)
        self.assertEqual(0.1, changed['data']['non_iid_alpha'])
        self.assertEqual('alpha_0.1', label)
        self.assertEqual(cfg['topology'], changed['topology'])


if __name__ == '__main__':
    unittest.main()
