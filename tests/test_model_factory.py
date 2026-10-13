import unittest


class ModelFactoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest('torch is not installed') from exc

    def test_simple_cnn_output_dimension(self):
        import torch
        from models.factory import build_model
        model = build_model({'model': {'name': 'simple_cnn'}}, 1, 10)
        self.assertEqual((2, 10), tuple(model(torch.randn(2, 1, 28, 28)).shape))

    def test_cifar_resnet18_output_dimension_and_batchnorm_state(self):
        import torch
        from models.factory import build_model
        model = build_model({'model': {'name': 'resnet18_cifar_bn'}}, 3, 100)
        self.assertEqual((2, 100), tuple(model(torch.randn(2, 3, 32, 32)).shape))
        state_keys = set(model.state_dict())
        self.assertTrue(any('running_mean' in key for key in state_keys))
        self.assertTrue(any('running_var' in key for key in state_keys))
        self.assertTrue(any('num_batches_tracked' in key for key in state_keys))


if __name__ == '__main__':
    unittest.main()
