"""Byzantine candidate-model attacks for the SLTrust-DFL protocol.

Each ``attack()`` receives a candidate model vector and returns the vector to
broadcast.  Attack classes set ``trains_byzantine_locally`` when a Byzantine
client must first produce a fresh local-SGD candidate; the trainer never lets
Byzantine clients aggregate neighbors.
"""
from __future__ import annotations

import torch


class NoAttack:
    name = "NoAttack"
    trains_byzantine_locally = False

    def attack(self, own_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        return own_vec


class BrokenNodeAttack:
    """Stale-node fault: retain and broadcast the old state."""
    name = "BrokenNode"
    trains_byzantine_locally = False


class GaussianAttack:
    """Additive model-space Gaussian perturbation."""
    name = "Gaussian"
    trains_byzantine_locally = False

    def __init__(self, std: float = 10.0):
        self.std = std

    def attack(self, own_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        return own_vec + torch.randn_like(own_vec) * self.std


class _TimedGaussianAttack:
    """Shared schedule logic for attacks that perturb fresh SGD candidates."""
    name = "TimedGaussian"
    trains_byzantine_locally = True

    def __init__(self, std: float = 10.0):
        self.std = std

    def _noise(self, candidate: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
        return candidate + torch.randn_like(candidate) * (self.std * multiplier)

    def participates_in_warmup_aggregation(self, t: int) -> bool:
        """Whether a benign warm-up candidate follows the honest update path."""
        return False


class BurnInAttack(_TimedGaussianAttack):
    """Benign SGD candidates until ``warmup_rounds``, then Gaussian candidates."""
    name = "BurnInGaussian"

    def __init__(self, warmup_rounds: int = 40, std: float = 10.0):
        super().__init__(std=std)
        self.warmup_rounds = warmup_rounds

    def participates_in_warmup_aggregation(self, t: int) -> bool:
        return t < self.warmup_rounds

    def attack(self, candidate: torch.Tensor, t: int = 0, **kwargs) -> torch.Tensor:
        return candidate if t < self.warmup_rounds else self._noise(candidate)


class OscillatingAttack(_TimedGaussianAttack):
    """Benign warm-up phase, then alternating Gaussian-perturbed and benign candidates.

    warmup_rounds benign rounds first (to accumulate reputation), then attack/rest cycles.
    """
    name = "OscillatingGaussian"

    def __init__(self, period_attack: int = 10, period_rest: int = 15, std: float = 10.0,
                 warmup_rounds: int = 0):
        super().__init__(std=std)
        self.period_attack = period_attack
        self.period_rest = period_rest
        self._period = period_attack + period_rest
        self.warmup_rounds = warmup_rounds

    def participates_in_warmup_aggregation(self, t: int) -> bool:
        return t < self.warmup_rounds

    def attack(self, candidate: torch.Tensor, t: int = 0, **kwargs) -> torch.Tensor:
        if t < self.warmup_rounds:
            return candidate
        phase = (t - self.warmup_rounds) % self._period
        if phase < self.period_attack:
            return self._noise(candidate)
        return candidate


class GradualGaussianAttack(_TimedGaussianAttack):
    """Ramp Gaussian perturbation from benign candidates after a warm-up phase."""
    name = "GradualGaussian"

    def __init__(self, warmup_rounds: int = 40, ramp_rounds: int = 20, std: float = 10.0):
        super().__init__(std=std)
        self.warmup_rounds = warmup_rounds
        self.ramp_rounds = ramp_rounds

    def participates_in_warmup_aggregation(self, t: int) -> bool:
        return t < self.warmup_rounds

    def attack(self, candidate: torch.Tensor, t: int = 0, **kwargs) -> torch.Tensor:
        if t < self.warmup_rounds:
            return candidate
        multiplier = min(1.0, (t - self.warmup_rounds) / max(self.ramp_rounds, 1))
        return self._noise(candidate, multiplier)


ATTACK_REGISTRY = {
    "none": NoAttack,
    "gaussian": GaussianAttack,
    "broken_node": BrokenNodeAttack,
    "burn_in": BurnInAttack,
    "oscillating": OscillatingAttack,
    "gradual_gaussian": GradualGaussianAttack,
}


def get_attack(name: str, **kwargs):
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](**kwargs)
