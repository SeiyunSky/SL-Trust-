"""Result identity, validation, and atomic persistence for experiment jobs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = 3
PROTOCOL = "candidate-model-state-v3"
# Separates protocol-compatible but semantically stale runs (old attack contract
# or S/L window) from formal results produced after the unified experiment plan.
EXPERIMENT_CONTRACT = "sltrust-dfl-2026-10-current-attacks-sl10-l30"
_REQUIRED_SERIES = ("test_acc", "test_loss", "consensus_error", "tau_byz")
_FINITE_SERIES = ("test_acc", "test_loss")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _effective_config(cfg: dict, method: str, attack: str) -> dict:
    """Keep only configuration that can affect this concrete training job."""
    common_keys = ("experiment", "federated", "data", "model", "topology", "optimizer", "logging")
    effective = {key: cfg[key] for key in common_keys if key in cfg}
    if method in cfg:
        effective["method_config"] = cfg[method]
    elif method == "sltrust_median" and "sltrust" in cfg:
        effective["method_config"] = dict(cfg["sltrust"], mode="median_bg")
    attack_configs = cfg.get("attack", {})
    effective["attack_config"] = attack_configs.get(attack, {})
    return effective


def result_identity(cfg: dict, suite: str, method: str, attack: str,
                    extra: dict | None = None) -> dict:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "experiment_contract": EXPERIMENT_CONTRACT,
        "suite": suite,
        "method": method,
        "attack": attack,
        "config": _effective_config(cfg, method, attack),
        "extra": extra or {},
    }
    payload["fingerprint"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def metadata_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + ".meta.json")


def expected_points(cfg: dict) -> int:
    rounds = int(cfg["federated"]["num_rounds"])
    interval = int(cfg["logging"]["log_interval"])
    return rounds // interval


def history_is_complete(history: object, cfg: dict) -> bool:
    if not isinstance(history, dict):
        return False
    n = expected_points(cfg)
    for key in _REQUIRED_SERIES:
        values = history.get(key)
        if not isinstance(values, list) or len(values) != n:
            return False
        if key in _FINITE_SERIES:
            try:
                if not all(math.isfinite(float(value)) for value in values):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def load_cached_result(result_path: Path, cfg: dict, suite: str, method: str,
                       attack: str, extra: dict | None = None) -> dict | None:
    """Return a complete matching result, never an unverified legacy JSON."""
    if os.environ.get("MVP_FORCE", "0") == "1":
        return None
    meta_path = metadata_path(result_path)
    if not result_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        expected = result_identity(cfg, suite, method, attack, extra)
        if metadata.get("fingerprint") != expected["fingerprint"]:
            return None
        with open(result_path, encoding="utf-8") as f:
            history = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return history if history_is_complete(history, cfg) else None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_result(result_path: Path, history: dict, cfg: dict, suite: str,
                method: str, attack: str, extra: dict | None = None) -> None:
    if not history_is_complete(history, cfg):
        raise ValueError(
            f"Refusing to save incomplete result {result_path}: "
            f"expected {expected_points(cfg)} checkpoints"
        )
    metadata = result_identity(cfg, suite, method, attack, extra)
    _atomic_json(result_path, history)
    _atomic_json(metadata_path(result_path), metadata)
