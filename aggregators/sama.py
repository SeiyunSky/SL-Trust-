"""
SAMA Aggregator — 单轮方向-幅度信任（无跨轮记忆）
复用自第一篇论文，作为 SLTrust 的对比基线
"""
import torch
from .base import BaseAggregator


class SAMAAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, eps=1e-8):
        super().__init__(name="SAMA", alpha=alpha)
        self.eps = eps

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
        w_i_norm = torch.norm(own_vec)
        if w_i_norm < self.eps:
            if return_stats:
                return own_vec, {'num_neighbors': N, 'avg_trust': 0.0}
            return own_vec

        neighbor_norms = torch.norm(neighbor_mat, dim=1)
        valid_mask = neighbor_norms >= self.eps

        cos_sims = torch.zeros(N, device=own_vec.device)
        if valid_mask.any():
            dots = (neighbor_mat[valid_mask] * own_vec).sum(dim=1)
            cos_sims[valid_mask] = dots / (neighbor_norms[valid_mask] * w_i_norm)

        phi = torch.clamp(cos_sims, min=0.0)
        valid = (phi > 0) & valid_mask

        if not valid.any():
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_filtered': N, 'avg_trust': 0.0}
            return own_vec

        valid_vecs = neighbor_mat[valid]
        valid_norms = neighbor_norms[valid]
        valid_phi = phi[valid]

        # 幅度对齐: w_i_norm / ||w_j|| * w_j
        aligned = w_i_norm * (valid_vecs / valid_norms.unsqueeze(1))
        total_weight = valid_phi.sum() + self.eps
        agg_vec = (valid_phi.unsqueeze(1) * aligned).sum(dim=0) / total_weight

        if return_stats:
            return agg_vec, {
                'num_neighbors': N,
                'num_filtered': N - int(valid.sum()),
                'avg_trust': valid_phi.mean().item(),
            }
        return agg_vec
