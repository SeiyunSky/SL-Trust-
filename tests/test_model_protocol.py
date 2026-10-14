import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed") from exc

from aggregators.fedavg import FedAvgAggregator
from aggregators.krum import KrumAggregator
from aggregators.sltrust import SLTrustAggregator
from aggregators.trimmed_mean import TrimmedMeanAggregator
from attacks import (BrokenNodeAttack, BurnInAttack, DirectionPerturbationAttack,
                     GaussianAttack, GradualGaussianAttack, LabelFlippingAttack,
                     OscillatingAttack)


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

    def test_gaussian_attacks_candidate_with_configured_std(self):
        candidate_model = torch.zeros(100)
        torch.manual_seed(7)
        attacked = GaussianAttack(std=2.0).attack(candidate_model)

        noise_std = (attacked - candidate_model).std()
        self.assertGreater(noise_std.item(), 1.5)
        self.assertLess(noise_std.item(), 2.5)

    def test_label_flipping_changes_labels_and_requires_fresh_candidate(self):
        attack = LabelFlippingAttack(num_classes=10)
        labels = torch.tensor([0, 1, 8, 9])

        torch.testing.assert_close(attack.flip_labels(labels), torch.tensor([9, 8, 1, 0]))
        self.assertTrue(attack.trains_byzantine_locally)

    def test_broken_node_keeps_stale_state_without_local_training(self):
        attack = BrokenNodeAttack()
        stale_model = torch.tensor([101.0, 98.0])

        torch.testing.assert_close(attack.attack(stale_model), stale_model)
        self.assertFalse(attack.trains_byzantine_locally)

    def test_timed_gaussian_attacks_use_fresh_candidate_and_schedule(self):
        candidate = torch.ones(16)
        burn_in = BurnInAttack(warmup_rounds=2, std=1.0)
        oscillating = OscillatingAttack(period_attack=1, period_rest=1, std=1.0)
        gradual = GradualGaussianAttack(warmup_rounds=2, ramp_rounds=2, std=1.0)

        torch.testing.assert_close(burn_in.attack(candidate, t=1), candidate)
        torch.testing.assert_close(oscillating.attack(candidate, t=1), candidate)
        torch.testing.assert_close(gradual.attack(candidate, t=2), candidate)
        self.assertFalse(torch.equal(burn_in.attack(candidate, t=2), candidate))
        self.assertFalse(torch.equal(oscillating.attack(candidate, t=0), candidate))
        self.assertFalse(torch.equal(gradual.attack(candidate, t=3), candidate))
        self.assertTrue(burn_in.trains_byzantine_locally)

    def test_direction_perturbation_operates_on_fresh_candidate(self):
        candidate = torch.tensor([3.0, 4.0])
        attack = DirectionPerturbationAttack(lambda_bd=0.3, seed=7)

        self.assertTrue(attack.trains_byzantine_locally)
        self.assertFalse(torch.equal(attack.attack(candidate), candidate))

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
