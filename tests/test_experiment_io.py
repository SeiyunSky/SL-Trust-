import json
import math
import tempfile
import unittest
from pathlib import Path

from experiment_io import (
    EXPERIMENT_CONTRACT,
    history_is_complete,
    load_cached_result,
    result_identity,
    save_result,
)


class ExperimentIOTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'experiment': {'seed': 42, 'device': 'cpu'},
            'federated': {'num_rounds': 10},
            'data': {'dataset': 'mnist'},
            'model': {'name': 'simple_cnn'},
            'topology': {'type': 'mesh', 'degree': 6},
            'optimizer': {'lr': 0.01},
            'logging': {'log_interval': 2},
            'sltrust': {'alpha': 0.5, 'beta': 0.1, 'S': 3, 'L': 10},
            'attack': {
                'burn_in': {'warmup_rounds': 4, 'scale': 1.0},
                'gaussian': {'std': 5.0},
            },
        }
        self.history = {
            'test_acc': [1, 2, 3, 4, 5],
            'test_loss': [1, 2, 3, 4, 5],
            'consensus_error': [1, 2, 3, 4, 5],
            'tau_byz': [1, 2, 3, 4, 5],
        }

    def test_identity_marks_current_experiment_contract(self):
        identity = result_identity(self.cfg, 'main_table', 'sltrust', 'burn_in')
        self.assertEqual(EXPERIMENT_CONTRACT, identity['experiment_contract'])

    def test_complete_history_uses_actual_logged_rounds(self):
        self.assertTrue(history_is_complete(self.history, self.cfg))
        incomplete = dict(self.history, test_acc=[1, 2])
        self.assertFalse(history_is_complete(incomplete, self.cfg))

    def test_non_finite_core_metrics_are_incomplete(self):
        for key in ('test_acc', 'test_loss', 'consensus_error'):
            invalid = dict(self.history)
            invalid[key] = list(self.history[key])
            invalid[key][-1] = math.nan
            self.assertFalse(history_is_complete(invalid, self.cfg), key)

    def test_tau_nan_is_allowed_for_nodes_without_byzantine_neighbors(self):
        history = dict(self.history, tau_byz=[math.nan] * 5)
        self.assertTrue(history_is_complete(history, self.cfg))

    def test_unrelated_attack_change_does_not_invalidate_fingerprint(self):
        before = result_identity(self.cfg, 'main_table', 'sltrust', 'burn_in')
        changed = json.loads(json.dumps(self.cfg))
        changed['attack']['gaussian']['std'] = 99.0
        after = result_identity(changed, 'main_table', 'sltrust', 'burn_in')
        self.assertEqual(before['fingerprint'], after['fingerprint'])

    def test_model_change_invalidates_fingerprint(self):
        before = result_identity(self.cfg, 'main_table', 'sltrust', 'burn_in')
        changed = json.loads(json.dumps(self.cfg))
        changed['model']['name'] = 'resnet18_cifar_bn'
        after = result_identity(changed, 'main_table', 'sltrust', 'burn_in')
        self.assertNotEqual(before['fingerprint'], after['fingerprint'])

    def test_stress_extra_change_invalidates_fingerprint(self):
        first = result_identity(
            self.cfg, 'stress_non_iid', 'sltrust', 'burn_in',
            {'dimension': 'non_iid', 'value': 0.1})
        second = result_identity(
            self.cfg, 'stress_non_iid', 'sltrust', 'burn_in',
            {'dimension': 'non_iid', 'value': 0.5})
        self.assertNotEqual(first['fingerprint'], second['fingerprint'])

    def test_active_method_or_attack_change_invalidates_fingerprint(self):
        before = result_identity(self.cfg, 'main_table', 'sltrust', 'burn_in')
        method_changed = json.loads(json.dumps(self.cfg))
        method_changed['sltrust']['beta'] = 0.2
        attack_changed = json.loads(json.dumps(self.cfg))
        attack_changed['attack']['burn_in']['warmup_rounds'] = 6
        self.assertNotEqual(before['fingerprint'], result_identity(
            method_changed, 'main_table', 'sltrust', 'burn_in')['fingerprint'])
        self.assertNotEqual(before['fingerprint'], result_identity(
            attack_changed, 'main_table', 'sltrust', 'burn_in')['fingerprint'])

    def test_atomic_save_and_verified_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            save_result(path, self.history, self.cfg, 'main_table', 'sltrust', 'burn_in')
            loaded = load_cached_result(path, self.cfg, 'main_table', 'sltrust', 'burn_in')
            self.assertEqual(self.history, loaded)

            changed = json.loads(json.dumps(self.cfg))
            changed['experiment']['seed'] = 123
            self.assertIsNone(load_cached_result(
                path, changed, 'main_table', 'sltrust', 'burn_in'))

    def test_legacy_json_without_metadata_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.json'
            path.write_text(json.dumps(self.history), encoding='utf-8')
            self.assertIsNone(load_cached_result(
                path, self.cfg, 'main_table', 'sltrust', 'burn_in'))


if __name__ == '__main__':
    unittest.main()
