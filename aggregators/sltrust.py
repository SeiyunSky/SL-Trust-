"""
SLTrust Aggregator — 双窗口历史感知信誉聚合（本文方案）

算法（每轮 t，节点 i 对邻居 j；输入为本地训练后的候选模型）：
  Step 1: φ_j^t = ReLU(cos(w_i^{t,+}, ŵ_j^{t,+}))
  Step 2: avg_S_j^t  = mean(hist[-S:])    最近窗口
          avg_bg_j^t = mean(hist[:-S])    非重叠历史背景窗口
  Step 3: 若背景窗口为空，ratio_j^t = 1；否则
          ratio_j^t = min(1, avg_S / (avg_bg + ε))
          τ_j^1 = φ_j^1
          τ_j^t = β·φ_j^t + (1-β)·τ_j^{t-1}·ratio_j^t, t > 1
  Step 4: AGG_i = Σ τ_j·(‖w_i^{t,+}‖/‖ŵ_j^{t,+}‖)·ŵ_j^{t,+} / Στ_j
          w_i^{t+1} = α·w_i^{t,+} + (1-α)·AGG_i

消融变体（通过 mode 参数控制）：
  'full'        — 完整双窗口（默认）
  'no_ratio'    — ratio 恒为 1，退化为标准 EMA
  'short_only'  — 背景也用短窗口长度 S，长期历史无效
  'long_only'   — avg_S = avg_L，ratio 恒为 1（等价于 no_ratio，验证短窗口价值）
"""
from collections import deque
import torch
from .base import BaseAggregator


class SLTrustAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, beta=0.1, S=5, L=20, eps=1e-8, mode='full'):
        """
        mode: 'full' | 'no_ratio' | 'short_only' | 'long_only' | 'median_bg'
        """
        valid_modes = {'full', 'no_ratio', 'no_clip', 'adaptive', 'short_only', 'long_only', 'median_bg'}
        if not 1 <= S < L:
            raise ValueError(f"SLTrust requires 1 <= S < L, got S={S}, L={L}")
        if not 0 < beta <= 1:
            raise ValueError(f"SLTrust requires 0 < beta <= 1, got beta={beta}")
        if eps <= 0:
            raise ValueError(f"SLTrust requires eps > 0, got eps={eps}")
        if mode not in valid_modes:
            raise ValueError(f"Unknown SLTrust mode: {mode}")

        super().__init__(name=f"SLTrust-{mode}", alpha=alpha)
        self.beta = beta
        self.S = S
        self.L = L
        self.eps = eps
        self.mode = mode
        self._tau: dict[int, float] = {}
        self._phi_hist: dict[int, deque] = {}
        self._adaptive_S: dict[int, int] = {} if mode == 'adaptive' else {}

    def reset(self):
        self._tau.clear()
        self._phi_hist.clear()
        self._adaptive_S.clear()

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

        # Step 1: 计算 φ_j（向量化）
        cos_sim = torch.mv(neighbor_mat, own_vec) / (w_i_norm * neighbor_norms + self.eps)
        phi = torch.clamp(cos_sim, min=0.0)
        phi[neighbor_norms < self.eps] = 0.0

        # Step 2 & 3: 双窗口 + 信誉更新
        tau = torch.zeros(N, device=own_vec.device)
        ratios = []
        for idx, nid in enumerate(neighbor_ids):
            phi_val = phi[idx].item()

            if nid not in self._phi_hist:
                self._phi_hist[nid] = deque(maxlen=self.L)

            hist = self._phi_hist[nid]
            hist.append(phi_val)
            hist_list = list(hist)

            # ratio 计算根据 mode 决定
            if self.mode == 'no_ratio':
                ratio = 1.0
            elif self.mode == 'short_only':
                # 短窗口和长窗口都用 S 长度（无跨期记忆），ratio 可能有轻微波动。
                # 用非重叠的两段各取 S，模拟"只有短期记忆"：
                # recent = 最近 S 轮，background = 倒数第 S+1 到 2S 轮。
                recent = hist_list[-self.S:]
                background = hist_list[-2 * self.S:-self.S] if len(hist_list) >= 2 * self.S else []
                if not background:
                    ratio = 1.0
                else:
                    avg_S = sum(recent) / len(recent)
                    avg_bg = sum(background) / len(background)
                    ratio = min(1.0, avg_S / (avg_bg + self.eps))
            elif self.mode == 'long_only':
                # avg_S = avg_L，ratio 恒为 1（验证短窗口的价值）
                ratio = 1.0
            elif self.mode == 'no_clip':
                recent = hist_list[-self.S:]
                background = hist_list[:-self.S]
                if not background:
                    ratio = 1.0
                else:
                    avg_S = sum(recent) / len(recent)
                    avg_bg = sum(background) / len(background)
                    ratio = avg_S / (avg_bg + self.eps)
            elif self.mode == 'adaptive':
                if nid not in self._adaptive_S:
                    self._adaptive_S[nid] = self.S
                S_eff = self._adaptive_S[nid]
                recent = hist_list[-S_eff:]
                background = hist_list[:-S_eff] if len(hist_list) > S_eff else []
                if not background:
                    ratio = 1.0
                else:
                    avg_S = sum(recent) / len(recent)
                    avg_bg = sum(background) / len(background)
                    ratio = min(1.0, avg_S / (avg_bg + self.eps))
                if ratio >= 0.9:
                    self._adaptive_S[nid] = min(self.L - 1, S_eff + 1)
                else:
                    self._adaptive_S[nid] = max(2, S_eff // 2)
            elif self.mode == 'median_bg':
                # A3: median 背景窗口变体（SLTrust-M）
                recent = hist_list[-self.S:]
                background = hist_list[:-self.S]
                if not background:
                    ratio = 1.0
                else:
                    import statistics as _stats
                    avg_S = sum(recent) / len(recent)
                    avg_bg = _stats.median(background)
                    ratio = min(1.0, avg_S / (avg_bg + self.eps))
            else:  # 'full'
                recent = hist_list[-self.S:]
                background = hist_list[:-self.S]
                if not background:
                    # 尚无独立历史基线时不作突变惩罚，退化为标准 EMA。
                    ratio = 1.0
                else:
                    avg_S = sum(recent) / len(recent)
                    avg_bg = sum(background) / len(background)
                    ratio = min(1.0, avg_S / (avg_bg + self.eps))

            ratios.append(ratio)
            # A1: 冷启动修复——第1轮直接用 phi_val 初始化，跳过 EMA 冷启
            if nid not in self._tau:
                tau_new = phi_val
            else:
                tau_prev = self._tau[nid]
                tau_new = self.beta * phi_val + (1 - self.beta) * tau_prev * ratio
            self._tau[nid] = tau_new
            tau[idx] = tau_new

        valid = (tau > 0) & (neighbor_norms >= self.eps)

        if not valid.any():
            if return_stats:
                return own_vec, {
                    'num_neighbors': N, 'avg_trust': 0.0, 'avg_ratio': 0.0,
                    # Retain per-edge quantities for trainer-side mechanism traces.
                    'neighbor_ids': list(neighbor_ids), 'phi': phi.detach().cpu().tolist(),
                    'ratio': list(ratios), 'tau': tau.detach().cpu().tolist(),
                }
            return own_vec

        # Step 4: 幅度对齐聚合
        valid_vecs  = neighbor_mat[valid]
        valid_norms = neighbor_norms[valid]
        valid_tau   = tau[valid]

        aligned = w_i_norm * (valid_vecs / valid_norms.unsqueeze(1))
        agg_vec = (valid_tau.unsqueeze(1) * aligned).sum(dim=0) / valid_tau.sum()

        if return_stats:
            return agg_vec, {
                'num_neighbors': N,
                'avg_trust':  tau.mean().item(),
                'min_tau':    tau.min().item(),
                'max_tau':    tau.max().item(),
                'avg_ratio':  sum(ratios) / len(ratios) if ratios else 1.0,
                # These arrays align with neighbor_ids and are intentionally raw:
                # trainer persists them as a separate Byzantine/honest edge trace.
                'neighbor_ids': list(neighbor_ids),
                'phi': phi.detach().cpu().tolist(),
                'ratio': list(ratios),
                'tau': tau.detach().cpu().tolist(),
            }
        return agg_vec
