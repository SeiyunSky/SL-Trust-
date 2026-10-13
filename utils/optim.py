"""Optimizer and round-level scheduler construction."""
import torch


def build_optimizer(parameters, config: dict):
    optimizer_cfg = config['optimizer']
    name = optimizer_cfg.get('name', 'sgd').lower()
    if name != 'sgd':
        raise ValueError(f"Unknown optimizer: {name}")
    return torch.optim.SGD(
        parameters,
        lr=optimizer_cfg['lr'],
        momentum=optimizer_cfg.get('momentum', 0.0),
        weight_decay=optimizer_cfg.get('weight_decay', 0.0),
    )


def build_scheduler(optimizer, config: dict):
    optimizer_cfg = config['optimizer']
    name = optimizer_cfg.get('scheduler', 'none').lower()
    if name == 'none':
        return None
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['federated']['num_rounds'])
    raise ValueError(f"Unknown scheduler: {name}")
