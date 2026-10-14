"""
CONTRA-Hard Aggregator — EMA 信誉 + 硬截断
τ < threshold 时直接归零，比 CONTRA（软 EMA）更接近原论文的硬过滤设计。
用于验证 SLTrust 优于"软/硬两种 EMA 基线"，加强主表结论。
"""
import torch
from .base import BaseAggregator


class CONTRAHardAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, beta=0.1, threshold=0.1, eps=1e-8):
        super().__init__(name="CONTRA-Hard", alpha=alpha)
        self.beta = beta
        self.threshold = threshold
        self.eps = eps
        self._tau: dict[int, float] = {}

    def reset(self):
        self._tau.clear()

    def aggregate(self, own_vec, neighbor_vecs, neighbor_ids=None,
                  t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'avg_trust': 0.0}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        N = neighbor_mat.shape[0]
        if neighbor_ids is None:
            neighbor_ids = list(range(N))

        w_i_norm = torch.norm(own_vec)
        if w_i_norm < self.eps:
            if return_stats:
                return own_vec, {'num_neighbors': N, 'avg_trust': 0.0}
            return own_vec

        neighbor_norms = torch.norm(neighbor_mat, dim=1)

        cos_sim = torch.mv(neighbor_mat, own_vec) / (w_i_norm * neighbor_norms + self.eps)
        phi = torch.clamp(cos_sim, min=0.0)
        phi[neighbor_norms < self.eps] = 0.0

        tau = torch.zeros(N, device=own_vec.device)
        for idx, nid in enumerate(neighbor_ids):
            # 冷启动修复：与 CONTRA / SLTrust 对齐
            if nid not in self._tau:
                tau_new = phi[idx].item()
            else:
                tau_new = self.beta * phi[idx].item() + (1 - self.beta) * self._tau[nid]
            # 硬截断：低于阈值归零
            if tau_new < self.threshold:
                tau_new = 0.0
            self._tau[nid] = tau_new
            tau[idx] = tau_new

        valid = (tau > 0) & (neighbor_norms >= self.eps)

        if not valid.any():
            if return_stats:
                return own_vec, {
                    'num_neighbors': N, 'avg_trust': 0.0,
                    'neighbor_ids': list(neighbor_ids),
                    'phi': phi.detach().cpu().tolist(),
                    'ratio': [1.0] * N,
                    'tau': tau.detach().cpu().tolist(),
                }
            return own_vec

        valid_vecs = neighbor_mat[valid]
        valid_norms = neighbor_norms[valid]
        valid_tau = tau[valid]

        aligned = w_i_norm * (valid_vecs / valid_norms.unsqueeze(1))
        agg_vec = (valid_tau.unsqueeze(1) * aligned).sum(dim=0) / valid_tau.sum()

        if return_stats:
            return agg_vec, {
                'num_neighbors': N,
                'avg_trust': tau.mean().item(),
                'min_tau': tau.min().item(),
                'max_tau': tau.max().item(),
                'neighbor_ids': list(neighbor_ids),
                'phi': phi.detach().cpu().tolist(),
                'ratio': [1.0] * N,
                'tau': tau.detach().cpu().tolist(),
            }
        return agg_vec
