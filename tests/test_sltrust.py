import math
import unittest

import torch

from aggregators.sltrust import SLTrustAggregator


class SLTrustAggregatorTest(unittest.TestCase):
    def setUp(self):
        self.own = torch.tensor([1.0, 0.0])

    @staticmethod
    def neighbor_with_cosine(value):
        return torch.tensor([value, math.sqrt(max(0.0, 1.0 - value * value))])

    def observe(self, aggregator, phi, neighbor_id=7):
        _, stats = aggregator.aggregate(
            self.own,
            [self.neighbor_with_cosine(phi)],
            neighbor_ids=[neighbor_id],
            return_stats=True,
        )
        return stats

    def test_rejects_invalid_parameters_and_mode(self):
        for kwargs in ({'S': 0, 'L': 3}, {'S': 3, 'L': 3}, {'S': 4, 'L': 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SLTrustAggregator(**kwargs)
        with self.assertRaises(ValueError):
            SLTrustAggregator(beta=0.0)
        with self.assertRaises(ValueError):
            SLTrustAggregator(eps=0.0)
        with self.assertRaises(ValueError):
            SLTrustAggregator(mode='unknown')

    def test_cold_start_uses_no_change_penalty(self):
        aggregator = SLTrustAggregator(beta=0.1, S=2, L=4)

        first = self.observe(aggregator, 0.8)
        second = self.observe(aggregator, 0.6)

        self.assertAlmostEqual(aggregator._tau[7], 0.1 * 0.6 + 0.9 * 0.8, places=6)
        self.assertAlmostEqual(first['avg_ratio'], 1.0, places=6)
        self.assertAlmostEqual(second['avg_ratio'], 1.0, places=6)

    def test_recent_and_background_windows_do_not_overlap(self):
        aggregator = SLTrustAggregator(beta=0.1, S=2, L=4)
        self.observe(aggregator, 1.0)
        self.observe(aggregator, 1.0)
        stats = self.observe(aggregator, 0.0)

        # recent=[1, 0], background=[1], so ratio=0.5.  A long window
        # overlapping recent samples would instead produce ratio=0.75.
        self.assertAlmostEqual(stats['avg_ratio'], 0.5, places=6)

    def test_ratio_is_clipped_and_does_not_accelerate_recovery(self):
        aggregator = SLTrustAggregator(beta=0.1, S=2, L=4)
        self.observe(aggregator, 0.2)
        self.observe(aggregator, 0.2)
        stats = self.observe(aggregator, 1.0)

        self.assertAlmostEqual(stats['avg_ratio'], 1.0, places=6)

    def test_behavior_drop_reduces_trust_faster_than_plain_ema(self):
        sltrust = SLTrustAggregator(beta=0.1, S=2, L=4, mode='full')
        ema = SLTrustAggregator(beta=0.1, S=2, L=4, mode='no_ratio')

        for phi in (1.0, 1.0, 1.0, 0.0):
            self.observe(sltrust, phi)
            self.observe(ema, phi)

        self.assertLess(sltrust._tau[7], ema._tau[7])
        self.assertAlmostEqual(sltrust._tau[7], 0.45, places=6)
        self.assertAlmostEqual(ema._tau[7], 0.9, places=6)

    def test_neighbor_histories_are_isolated_and_reset_clears_state(self):
        aggregator = SLTrustAggregator(S=2, L=4)
        self.observe(aggregator, 1.0, neighbor_id=1)
        self.observe(aggregator, 0.2, neighbor_id=2)

        self.assertEqual(set(aggregator._phi_hist), {1, 2})
        self.assertNotEqual(aggregator._tau[1], aggregator._tau[2])

        aggregator.reset()
        self.assertEqual(aggregator._phi_hist, {})
        self.assertEqual(aggregator._tau, {})


if __name__ == '__main__':
    unittest.main()
