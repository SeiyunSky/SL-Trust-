"""Byzantine candidate-model attacks for the SLTrust-DFL protocol.

Each ``attack()`` receives a candidate model vector and returns the vector to
broadcast.  Attack classes set ``trains_byzantine_locally`` when a Byzantine
client must first produce a fresh local-SGD candidate; the trainer never lets
Byzantine clients aggregate neighbors.
"""
from __future__ import annotations

import numpy as np
import torch


class NoAttack:
    name = "NoAttack"
    trains_byzantine_locally = False

    def attack(self, own_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        return own_vec


class BrokenNodeAttack:
    """SAMA-aligned stale-node fault: retain and broadcast the old state."""
    name = "BrokenNode"
    trains_byzantine_locally = False

    def attack(self, own_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        return own_vec


class GaussianAttack:
    """SAMA-aligned additive model-space Gaussian perturbation."""
    name = "Gaussian"
    trains_byzantine_locally = False

    def __init__(self, std: float = 10.0):
        self.std = std

    def attack(self, own_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        return own_vec + torch.randn_like(own_vec) * self.std


class LabelFlippingAttack:
    """SAMA-aligned data-poisoning attack with ``y -> C - 1 - y``."""
    name = "LabelFlipping"
    trains_byzantine_locally = True

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return (self.num_classes - 1) - labels

    def attack(self, own_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        return own_vec


class ALIEAttack:
    """SAMA-aligned omniscient A Little Is Enough candidate construction."""
    name = "ALIE"
    trains_byzantine_locally = False

    def __init__(self, num_byzantine: int, num_clients: int):
        self.num_byzantine = num_byzantine
        self.num_clients = num_clients

    def attack(self, honest_vecs, **kwargs) -> torch.Tensor:
        mat = torch.stack(honest_vecs) if isinstance(honest_vecs, list) else honest_vecs
        n, b = self.num_clients, self.num_byzantine
        s = int(np.floor(n / 2.0 + 1)) - b
        from scipy import stats as scipy_stats

        threshold = np.clip((n - b - s) / (n - b), 1e-6, 1 - 1e-6)
        z = float(scipy_stats.norm.ppf(threshold))
        return mat.mean(dim=0) - z * mat.std(dim=0).clamp(min=1e-6)


class _TimedGaussianAttack:
    """Shared schedule logic for attacks that perturb fresh SGD candidates."""
    name = "TimedGaussian"
    trains_byzantine_locally = True

    def __init__(self, std: float = 10.0):
        self.std = std

    def _noise(self, candidate: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
        return candidate + torch.randn_like(candidate) * (self.std * multiplier)


class BurnInAttack(_TimedGaussianAttack):
    """Benign SGD candidates until ``warmup_rounds``, then Gaussian candidates."""
    name = "BurnInGaussian"

    def __init__(self, warmup_rounds: int = 40, std: float = 10.0):
        super().__init__(std=std)
        self.warmup_rounds = warmup_rounds

    def attack(self, candidate: torch.Tensor, t: int = 0, **kwargs) -> torch.Tensor:
        return candidate if t < self.warmup_rounds else self._noise(candidate)


class OscillatingAttack(_TimedGaussianAttack):
    """Alternate Gaussian-perturbed and benign fresh SGD candidates."""
    name = "OscillatingGaussian"

    def __init__(self, period_attack: int = 10, period_rest: int = 15, std: float = 10.0):
        super().__init__(std=std)
        self.period_attack = period_attack
        self.period_rest = period_rest
        self._period = period_attack + period_rest

    def attack(self, candidate: torch.Tensor, t: int = 0, **kwargs) -> torch.Tensor:
        if t % self._period < self.period_attack:
            return self._noise(candidate)
        return candidate


class GradualGaussianAttack(_TimedGaussianAttack):
    """Ramp Gaussian perturbation from benign candidates after a warm-up phase."""
    name = "GradualGaussian"

    def __init__(self, warmup_rounds: int = 40, ramp_rounds: int = 20, std: float = 10.0):
        super().__init__(std=std)
        self.warmup_rounds = warmup_rounds
        self.ramp_rounds = ramp_rounds

    def attack(self, candidate: torch.Tensor, t: int = 0, **kwargs) -> torch.Tensor:
        if t < self.warmup_rounds:
            return candidate
        multiplier = min(1.0, (t - self.warmup_rounds) / max(self.ramp_rounds, 1))
        return self._noise(candidate, multiplier)


class DirectionPerturbationAttack:
    """Continuous fixed-direction perturbation of each fresh SGD candidate.

    This is a model-space boundary test, not a data-trigger backdoor attack.
    """
    name = "DirectionPerturbation"
    trains_byzantine_locally = True

    def __init__(self, lambda_bd: float = 0.3, seed: int = 0):
        self.lambda_bd = lambda_bd
        self._seed = seed
        self._trigger = None

    def _get_trigger(self, vec: torch.Tensor) -> torch.Tensor:
        if self._trigger is None or self._trigger.shape != vec.shape:
            generator = torch.Generator(device=vec.device)
            generator.manual_seed(self._seed)
            trigger = torch.randn(vec.shape, generator=generator, device=vec.device)
            self._trigger = trigger / (trigger.norm() + 1e-8)
        return self._trigger.to(vec.device)

    def attack(self, candidate: torch.Tensor, **kwargs) -> torch.Tensor:
        trigger = self._get_trigger(candidate)
        return ((1.0 - self.lambda_bd) * candidate
                + self.lambda_bd * trigger * candidate.norm())


ATTACK_REGISTRY = {
    "none": NoAttack,
    "gaussian": GaussianAttack,
    "label_flipping": LabelFlippingAttack,
    "alie": ALIEAttack,
    "broken_node": BrokenNodeAttack,
    "burn_in": BurnInAttack,
    "oscillating": OscillatingAttack,
    "gradual_gaussian": GradualGaussianAttack,
    "direction_perturbation": DirectionPerturbationAttack,
}


def get_attack(name: str, **kwargs):
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](**kwargs)
