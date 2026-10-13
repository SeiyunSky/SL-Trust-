import unittest


class DataPartitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np  # noqa: F401
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest('data dependencies are not installed') from exc

    def test_cifar100_style_partition_is_nonempty_unique_and_reproducible(self):
        import numpy as np
        from utils.data_loader import dirichlet_partition

        class Dataset:
            targets = [label for label in range(100) for _ in range(20)]

        np.random.seed(42)
        first = dirichlet_partition(Dataset(), 20, alpha=0.1, num_classes=100)
        np.random.seed(42)
        second = dirichlet_partition(Dataset(), 20, alpha=0.1, num_classes=100)
        self.assertEqual(first, second)
        self.assertTrue(all(first))
        flattened = [index for indices in first for index in indices]
        self.assertEqual(len(Dataset.targets), len(flattened))
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == '__main__':
    unittest.main()
