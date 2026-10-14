"""Single-file experiment configuration loader."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "configs" / "experiments.yaml"
_LEGACY_PRESET = {
    "main_table.yaml": "main_table_mnist",
    "mvp.yaml": "mvp_mnist",
    "ablation.yaml": "ablation_mnist",
    "tau.yaml": "tau_mnist",
    "sensitivity.yaml": "sensitivity_mnist",
    "main_table_cifar10.yaml": "main_table_cifar10",
    "mvp_cifar10.yaml": "mvp_cifar10",
    "ablation_cifar10.yaml": "ablation_cifar10",
    "tau_cifar10.yaml": "tau_cifar10",
    "generalization_cifar100.yaml": "generalization_cifar100",
    "stress_non_iid_mnist.yaml": "stress_non_iid_mnist",
    "stress_topology_mnist.yaml": "stress_topology_mnist",
    "stress_byzantine_ratio_mnist.yaml": "stress_byzantine_ratio_mnist",
}


def _merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_experiment_config(preset: str | Path) -> dict:
    """Load one named preset from the sole experiment YAML file."""
    preset = _LEGACY_PRESET.get(Path(str(preset)).name, str(preset))
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    presets = document["presets"]
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {sorted(presets)}")

    cfg = _merge(document["defaults"], presets[preset])
    method_params = _merge(document["method_params"], cfg.pop("method_overrides", {}))
    cfg.update(method_params)
    cfg["attack"] = _merge(document["attack_params"], cfg.pop("attack_overrides", {}))
    return cfg
