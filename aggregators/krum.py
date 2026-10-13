"""Krum / Multi-Krum Aggregator"""
import torch
from .base import BaseAggregator


class KrumAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, multi_k=None, byzantine_ratio=0.2):
        super().__init__(name="Multi-Krum" if multi_k else "Krum", alpha=alpha)
        self.multi_k = multi_k
        self.byzantine_ratio = byzantine_ratio

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'avg_trust': None}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        # Krum 只在收到的邻居候选模型中选择；own_vec 由
        # final_update(alpha) 统一作为 self-anchor 注入一次。
        vecs = neighbor_mat
        n = vecs.shape[0]
        f = max(0, int(n * self.byzantine_ratio))
        # Krum requires n >= 2f + 3. A global Byzantine ratio does not ensure
        # that condition in a small DFL neighborhood, so fail explicitly.
        if n < 2 * f + 3:
            raise ValueError(f"Krum requires n >= 2f + 3, got n={n}, f={f}")
        k = n - f - 2

        diff = vecs.unsqueeze(0) - vecs.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=2)
        dist_sq.fill_diagonal_(float('inf'))
        topk_vals, _ = torch.topk(dist_sq, k, dim=1, largest=False)
        scores = topk_vals.sum(dim=1)

        if self.multi_k is not None:
            m = max(1, min(self.multi_k, n - f))
            selected_idx = torch.topk(scores, m, largest=False).indices
            agg_vec = vecs[selected_idx].mean(dim=0)
        else:
            agg_vec = vecs[scores.argmin()]

        if return_stats:
            return agg_vec, {'num_neighbors': n, 'avg_trust': None}
        return agg_vec
