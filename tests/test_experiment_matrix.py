import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from experiment_matrix import apply_stress_value, main_table_tasks, stress_tasks


ROOT = Path(__file__).parent.parent


class ExperimentMatrixTest(unittest.TestCase):
    def _load(self, name):
        with open(ROOT / 'configs' / name, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def test_cifar100_matrix_has_exactly_48_unique_tasks(self):
        tasks = main_table_tasks(self._load('generalization_cifar100.yaml'))
        keys = {(t['seed'], t['method'], t['attack']) for t in tasks}
        self.assertEqual(48, len(tasks))
        self.assertEqual(len(tasks), len(keys))

    def test_configured_seed_runner_executes_every_seed(self):
        from experiments.performance import main_table_experiment

        seen = []
        with patch.object(main_table_experiment, 'run_main_table',
                          side_effect=lambda _: seen.append(os.environ['MVP_SEED'])):
            main_table_experiment.run_configured_seeds(
                ROOT / 'configs' / 'generalization_cifar100.yaml')
        self.assertEqual(['42', '123', '456'], seen)

    def test_stress_task_counts_are_unique(self):
        expected = {
            'stress_non_iid_mnist.yaml': 81,
            'stress_topology_mnist.yaml': 81,
            'stress_byzantine_ratio_mnist.yaml': 54,
        }
        for name, count in expected.items():
            tasks = stress_tasks(self._load(name))
            keys = {
                (t['seed'], t['method'], t['attack'], repr(t['value']))
                for t in tasks
            }
            self.assertEqual(count, len(tasks), name)
            self.assertEqual(len(tasks), len(keys), name)

    def test_stress_value_only_changes_selected_dimension(self):
        cfg = self._load('stress_non_iid_mnist.yaml')
        changed, label = apply_stress_value(cfg, 0.1)
        self.assertEqual(0.1, changed['data']['non_iid_alpha'])
        self.assertEqual('alpha_0.1', label)
        self.assertEqual(cfg['topology'], changed['topology'])


if __name__ == '__main__':
    unittest.main()
