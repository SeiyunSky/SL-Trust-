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

        # Align the FedAvg baseline with SAMA: aggregate the receiver's local
        # candidate together with its received neighbor candidates.  The common
        # final_update() self-anchor remains applied afterwards.
        vecs = torch.cat([own_vec.unsqueeze(0), neighbor_mat], dim=0)
        agg_vec = vecs.mean(dim=0)

        if return_stats:
            return agg_vec, {'num_neighbors': neighbor_mat.shape[0], 'avg_trust': None}
        return agg_vec
