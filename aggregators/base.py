"""Base aggregator class for candidate-model aggregation."""
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch


class BaseAggregator(ABC):

    def __init__(self, name="BaseAggregator", alpha=0.5):
        self.name = name
        self.alpha = alpha

    @staticmethod
    def model_to_vector(model_or_state):
        if isinstance(model_or_state, torch.nn.Module):
            return torch.cat([p.data.reshape(-1).float()
                              for p in model_or_state.parameters()])
        vectors = []
        for key in sorted(model_or_state.keys()):
            param = model_or_state[key]
            if isinstance(param, torch.Tensor):
                vectors.append(param.reshape(-1).float())
        return torch.cat(vectors)

    @staticmethod
    def load_from_vector(model, vector):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(vector[idx:idx + n].reshape(p.shape))
            idx += n

    def final_update(self, own_vec, agg_vec, alpha=None, **kwargs):
        """融合本地候选模型与邻居聚合：w_next = α·w_local + (1-α)·AGG。"""
        if alpha is None:
            alpha = self.alpha
        return alpha * own_vec + (1 - alpha) * agg_vec

    @abstractmethod
    def aggregate(self, own_vec, neighbor_vecs, **kwargs):
        pass

    @staticmethod
    def compute_cosine_similarity(vec1, vec2, eps=1e-8):
        n1 = torch.norm(vec1)
        n2 = torch.norm(vec2)
        if n1 < eps or n2 < eps:
            return 0.0
        return (torch.dot(vec1, vec2) / (n1 * n2)).item()
