import unittest
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


class TopologyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import networkx  # noqa: F401
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("networkx is not installed") from exc

    def setUp(self):
        if np is None:
            self.skipTest("numpy is not installed")

    def test_mesh_has_exact_requested_degree_and_is_reproducible(self):
        from utils.topology import generate_mesh_topology

        np.random.seed(42)
        first = generate_mesh_topology(20, degree=4)
        np.random.seed(42)
        second = generate_mesh_topology(20, degree=4)

        self.assertEqual(first, second)
        self.assertTrue(all(len(neighbors) == 4 for neighbors in first))
        for node, neighbors in enumerate(first):
            self.assertNotIn(node, neighbors)
            for neighbor in neighbors:
                self.assertIn(node, first[neighbor])

    def test_explicit_seed_reproduces_regular_and_small_world_graphs(self):
        from utils.topology import generate_mesh_topology, generate_small_world_topology

        self.assertEqual(
            generate_mesh_topology(20, degree=6, seed=123),
            generate_mesh_topology(20, degree=6, seed=123))
        first = generate_small_world_topology(20, degree=6, rewire_prob=0.2, seed=123)
        second = generate_small_world_topology(20, degree=6, rewire_prob=0.2, seed=123)
        self.assertEqual(first, second)
        self.assertTrue(all(first))

    def test_mesh_rejects_impossible_regular_graph(self):
        from utils.topology import generate_mesh_topology

        with self.assertRaises(ValueError):
            generate_mesh_topology(5, degree=3)

    def test_seed_is_set_before_data_and_topology_generation(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("torch is not installed") from exc

        import trainer

        cfg = {
            'experiment': {'device': 'cpu', 'seed': 123},
            'federated': {'num_clients': 4, 'byzantine_ratio': 0.25, 'batch_size': 8},
            'data': {'dataset': 'mnist', 'data_dir': './data', 'non_iid_alpha': 0.5},
            'topology': {'type': 'mesh', 'degree': 2},
        }
        seen = {}

        def fake_loader(config, device):
            seen['loader_numpy'] = int(np.random.randint(1_000_000))
            seen['loader_torch'] = int(torch.randint(1_000_000, (1,)).item())
            return [object()] * 4, object(), {'in_channels': 1, 'num_classes': 10}

        def fake_topology(config, n, seed):
            seen['topology_numpy'] = int(np.random.randint(1_000_000))
            seen['topology_seed'] = seed
            return [[1, 3], [0, 2], [1, 3], [0, 2]]

        with patch.object(trainer, 'load_federated_dataset', side_effect=fake_loader), \
             patch.object(trainer, 'generate_topology', side_effect=fake_topology):
            trainer.FederatedTrainer(cfg)
            first = dict(seen)
            trainer.FederatedTrainer(cfg)
            second = dict(seen)

        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
