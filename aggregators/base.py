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
        """Serialize the complete state, including BatchNorm running statistics."""
        state = (model_or_state.state_dict() if isinstance(model_or_state, torch.nn.Module)
                 else model_or_state)
        vectors = [tensor.detach().reshape(-1).float() for tensor in state.values()
                   if isinstance(tensor, torch.Tensor)]
        if not vectors:
            raise ValueError("Model state contains no tensors")
        return torch.cat(vectors)

    @staticmethod
    def load_from_vector(model, vector):
        """Restore parameters and buffers in state_dict order and original dtypes."""
        state = model.state_dict()
        restored = OrderedDict()
        idx = 0
        for key, tensor in state.items():
            n = tensor.numel()
            values = vector[idx:idx + n]
            if values.numel() != n:
                raise ValueError(
                    f"State vector is too short while restoring {key}: "
                    f"need {n}, got {values.numel()}")
            values = values.reshape(tensor.shape)
            if key.endswith('running_var'):
                values = values.clamp_min(0.0)
            if not tensor.is_floating_point() and not tensor.is_complex():
                values = values.round().clamp_min(0)
            restored[key] = values.to(device=tensor.device, dtype=tensor.dtype)
            idx += n
        if idx != vector.numel():
            raise ValueError(
                f"State vector has {vector.numel() - idx} trailing values")
        model.load_state_dict(restored, strict=True)

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
