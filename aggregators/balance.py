"""BALANCE Aggregator"""
import torch
import numpy as np
from .base import BaseAggregator


class BALANCEAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, gamma=0.5, kappa=0.1):
        super().__init__(name="BALANCE", alpha=alpha)
        self.gamma = gamma
        self.kappa = kappa

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'avg_trust': 0.0}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        N = neighbor_mat.shape[0]
        w_i_norm = torch.norm(own_vec).item()
        decay = np.exp(-self.kappa * t / T)
        threshold = self.gamma * decay * w_i_norm

        dists = torch.norm(neighbor_mat - own_vec.unsqueeze(0), dim=1)
        accept_mask = dists <= threshold

        if not accept_mask.any():
            if return_stats:
                return own_vec, {'num_neighbors': N, 'avg_trust': 0.0}
            return own_vec

        agg_vec = neighbor_mat[accept_mask].mean(dim=0)

        if return_stats:
            return agg_vec, {'num_neighbors': N, 'avg_trust': float(accept_mask.float().mean())}
        return agg_vec
