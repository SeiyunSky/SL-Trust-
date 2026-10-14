"""
CONTRA Aggregator — DFL 场景普通 EMA 信誉（无双窗口）
作为 SLTrust 的消融对照：有历史记忆，但无 farming 防御

算法（输入均为本地训练后的候选模型）：
  τ_j^t = β·φ_j^t + (1-β)·τ_j^{t-1}       （标准 EMA，无 ratio 惩罚）
  AGG_i = Σ τ_j·(‖w_i'‖/‖ŵ_j'‖)·ŵ_j' / Στ_j
  w_i^{t+1} = α·w_i' + (1-α)·AGG_i
"""
import torch
from .base import BaseAggregator


class CONTRAAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, beta=0.1, eps=1e-8):
        super().__init__(name="CONTRA", alpha=alpha)
        self.beta = beta
        self.eps = eps
        # 每个节点维护自己对所有邻居的信誉 {neighbor_id: tau}
        # 在 trainer 中按 node_id 调用，故这里用 dict 存
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

        # Step 1: 计算候选模型之间的 φ_j = ReLU(cos(w_i', ŵ_j'))
        cos_sim = torch.mv(neighbor_mat, own_vec) / (w_i_norm * neighbor_norms + self.eps)
        phi = torch.clamp(cos_sim, min=0.0)
        phi[neighbor_norms < self.eps] = 0.0

        # Step 2: 标准 EMA 信誉更新（无 ratio）
        tau = torch.zeros(N, device=own_vec.device)
        for idx, nid in enumerate(neighbor_ids):
            # 冷启动修复：第1轮直接用 phi_val，跳过 EMA 冷启，与 SLTrust 对齐
            if nid not in self._tau:
                tau_new = phi[idx].item()
            else:
                tau_new = self.beta * phi[idx].item() + (1 - self.beta) * self._tau[nid]
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

        # Step 3: 幅度对齐聚合
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
