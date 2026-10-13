"""Trimmed Mean Aggregator"""
import torch
from .base import BaseAggregator


class TrimmedMeanAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, trim_ratio=0.1):
        super().__init__(name="TrimmedMean", alpha=alpha)
        self.trim_ratio = trim_ratio

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'avg_trust': None}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        # 坐标截断只作用于邻居候选模型；own_vec 由
        # final_update(alpha) 统一作为 self-anchor 注入一次。
        vecs = neighbor_mat
        n = vecs.shape[0]
        # 小邻域中 floor(n * ratio) 常为 0，导致“Trimmed Mean”退化为均值。
        # 只要配置了正 trim_ratio，就至少截去两端各一个坐标值。
        k = max(1, int(n * self.trim_ratio)) if self.trim_ratio > 0 else 0
        if 2 * k >= n:
            k = 0

        sorted_vecs, _ = torch.sort(vecs, dim=0)
        trimmed = sorted_vecs[k:n - k] if k > 0 else sorted_vecs
        agg_vec = trimmed.mean(dim=0)

        if return_stats:
            return agg_vec, {'num_neighbors': n, 'avg_trust': None, 'trim_k': k}
        return agg_vec
