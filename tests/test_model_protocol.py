import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed") from exc

from aggregators.fedavg import FedAvgAggregator
from aggregators.krum import KrumAggregator
from aggregators.sltrust import SLTrustAggregator
from aggregators.trimmed_mean import TrimmedMeanAggregator
from attacks import GaussianAttack, SignFlipAttack


class ModelProtocolTest(unittest.TestCase):
    def test_complete_state_vector_roundtrip_includes_batchnorm_buffers(self):
        from models.resnet_cifar import CIFARResNet18

        source = CIFARResNet18(num_classes=10)
        source.train()
        source(torch.randn(4, 3, 32, 32))
        vector = FedAvgAggregator.model_to_vector(source)
        target = CIFARResNet18(num_classes=10)
        FedAvgAggregator.load_from_vector(target, vector)

        for key, tensor in source.state_dict().items():
            torch.testing.assert_close(tensor, target.state_dict()[key])

    def test_batchnorm_state_projection_keeps_attacked_state_valid(self):
        from models.resnet_cifar import CIFARResNet18

        model = CIFARResNet18(num_classes=10)
        vector = FedAvgAggregator.model_to_vector(model)
        FedAvgAggregator.load_from_vector(model, -vector)
        for key, tensor in model.state_dict().items():
            if key.endswith('running_var') or key.endswith('num_batches_tracked'):
                self.assertTrue(torch.all(tensor >= 0), key)

    def test_candidate_model_is_written_back_directly(self):
        own_model = torch.tensor([100.0, 100.0])
        neighbor_model = torch.tensor([102.0, 98.0])
        aggregator = FedAvgAggregator(alpha=0.5)

        aggregate_model = aggregator.aggregate(own_model, [neighbor_model])
        next_model = aggregator.final_update(own_model, aggregate_model)

        torch.testing.assert_close(aggregate_model, neighbor_model)
        torch.testing.assert_close(next_model, torch.tensor([101.0, 99.0]))

    def test_neighbor_aggregators_do_not_duplicate_self_model(self):
        own = torch.tensor([100.0, 100.0])
        neighbors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

        fedavg = FedAvgAggregator().aggregate(own, neighbors)
        trimmed = TrimmedMeanAggregator(trim_ratio=0.0).aggregate(own, neighbors)
        krum_neighbors = neighbors + [torch.tensor([2.0, 3.0])]
        krum = KrumAggregator(byzantine_ratio=0.0).aggregate(own, krum_neighbors)

        torch.testing.assert_close(fedavg, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(trimmed, torch.tensor([2.0, 3.0]))
        self.assertTrue(any(torch.equal(krum, neighbor) for neighbor in krum_neighbors))

    def test_krum_rejects_too_small_neighborhood(self):
        own = torch.tensor([0.0, 0.0])
        neighbors = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
                     torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])]

        with self.assertRaises(ValueError):
            KrumAggregator(byzantine_ratio=0.2).aggregate(own, neighbors)

    def test_sign_flip_attacks_candidate_model(self):
        candidate_model = torch.tensor([101.0, 98.0])

        malicious_model = SignFlipAttack(scale=1.0).attack(candidate_model)

        torch.testing.assert_close(malicious_model, torch.tensor([-101.0, -98.0]))

    def test_malicious_candidate_is_written_back_as_internal_state(self):
        internal_model = torch.tensor([101.0, 98.0])

        malicious_candidate = SignFlipAttack().attack(internal_model)
        next_internal_model = malicious_candidate

        torch.testing.assert_close(malicious_candidate, torch.tensor([-101.0, -98.0]))
        torch.testing.assert_close(next_internal_model, malicious_candidate)

    def test_gaussian_noise_is_scaled_by_candidate_model_rms(self):
        candidate_model = torch.full((100,), 0.01)
        torch.manual_seed(7)
        attacked = GaussianAttack(std=2.0).attack(candidate_model)

        noise_rms = (attacked - candidate_model).norm() / candidate_model.numel() ** 0.5
        self.assertGreater(noise_rms.item(), 0.01)
        self.assertLess(noise_rms.item(), 0.03)

    def test_sltrust_aligns_neighbor_model_to_own_norm(self):
        own = torch.tensor([3.0, 4.0])
        same_direction_larger_model = torch.tensor([30.0, 40.0])
        aggregator = SLTrustAggregator(alpha=0.5, beta=0.1, S=2, L=4)

        aggregate_model = aggregator.aggregate(
            own, [same_direction_larger_model], neighbor_ids=[9])
        next_model = aggregator.final_update(own, aggregate_model)

        torch.testing.assert_close(aggregate_model, own)
        torch.testing.assert_close(next_model, own)


if __name__ == '__main__':
    unittest.main()
