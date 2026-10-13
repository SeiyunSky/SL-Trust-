"""FedAvg Aggregator"""
import torch
from .base import BaseAggregator


class FedAvgAggregator(BaseAggregator):

    def __init__(self, alpha=0.5):
        super().__init__(name="FedAvg", alpha=alpha)

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'avg_trust': None}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        # aggregate() 只聚合邻居候选模型；own_vec 由
        # BaseAggregator.final_update() 通过 alpha 统一注入一次。
        agg_vec = neighbor_mat.mean(dim=0)

        if return_stats:
            return agg_vec, {'num_neighbors': neighbor_mat.shape[0], 'avg_trust': None}
        return agg_vec
