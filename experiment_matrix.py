"""Dependency-light experiment task enumeration."""
from __future__ import annotations

import copy
import json


def main_table_tasks(config: dict) -> list[dict]:
    matrix = config.get('matrix', {})
    methods = matrix.get('methods', [
        'fedavg', 'sama', 'balance', 'krum', 'trimmed_mean',
        'contra', 'contra_hard', 'sltrust'])
    attacks = matrix.get('attacks', [
        'none', 'gaussian', 'label_flipping', 'alie', 'broken_node',
        'burn_in', 'oscillating'])
    seeds = matrix.get('seeds', [config['experiment']['seed']])
    return [
        {'method': method, 'attack': attack, 'seed': int(seed)}
        for seed in seeds for attack in attacks for method in methods
    ]


def sensitivity_tasks(config: dict) -> list[dict]:
    """Enumerate YAML-defined attack-parameter sensitivity tasks."""
    sensitivity = config['sensitivity']
    tasks = []
    for seed in sensitivity['seeds']:
        for method in sensitivity['methods']:
            for warmup in sensitivity['burnin_warmups']:
                tasks.append({
                    'method': method, 'attack': 'burn_in', 'seed': int(seed),
                    'parameter': 'warmup', 'value': int(warmup),
                })
            for period in sensitivity['oscillating_periods']:
                tasks.append({
                    'method': method, 'attack': 'oscillating', 'seed': int(seed),
                    'parameter': 'period', 'value': int(period),
                })
            for lambda_bd in sensitivity['direction_lambdas']:
                tasks.append({
                    'method': method, 'attack': 'direction_perturbation',
                    'seed': int(seed), 'parameter': 'lambda',
                    'value': float(lambda_bd),
                })
    return tasks


def stress_tasks(config: dict) -> list[dict]:
    stress = config['stress']
    return [
        {'method': method, 'attack': attack, 'seed': int(seed), 'value': value}
        for seed in stress['seeds'] for value in stress['values']
        for attack in stress['attacks'] for method in stress['methods']
    ]


def apply_stress_value(config: dict, value) -> tuple[dict, str]:
    cfg = copy.deepcopy(config)
    dimension = cfg['stress']['dimension']
    if dimension == 'non_iid':
        cfg['data']['non_iid_alpha'] = float(value)
        label = f"alpha_{float(value):g}"
    elif dimension == 'byzantine_ratio':
        cfg['federated']['byzantine_ratio'] = float(value)
        label = f"ratio_{float(value):g}"
    elif dimension == 'topology':
        if not isinstance(value, dict) or 'type' not in value:
            raise ValueError(f"Invalid topology stress value: {value!r}")
        cfg['topology'] = copy.deepcopy(value)
        parts = [str(value['type'])]
        if 'degree' in value:
            parts.append(f"d{value['degree']}")
        if 'rewire_prob' in value:
            parts.append(f"p{value['rewire_prob']:g}")
        label = '_'.join(parts)
    else:
        raise ValueError(f"Unknown stress dimension: {dimension}")
    return cfg, label


def encode_value(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def decode_value(value: str):
    return json.loads(value)
