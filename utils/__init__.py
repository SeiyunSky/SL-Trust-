"""Lightweight utility exports; training dependencies stay lazily imported."""
from pathlib import Path

from .datasets import check_dataset, check_all_datasets, download_dataset


_DATA_LOADER_EXPORTS = {
    'load_mnist', 'load_mnist_gpu', 'load_cifar10', 'load_cifar10_gpu',
    'load_cifar100', 'load_federated_dataset', 'dirichlet_partition', 'GPUTensorLoader',
}
_OPTIM_EXPORTS = {'build_optimizer', 'build_scheduler'}
_TOPOLOGY_EXPORTS = {
    'generate_ring_topology', 'generate_mesh_topology', 'generate_small_world_topology',
    'generate_topology', 'check_connectivity',
}


def __getattr__(name):
    """Lazily provide legacy exports without loading Torch during TUI startup."""
    if name in _DATA_LOADER_EXPORTS:
        from . import data_loader
        return getattr(data_loader, name)
    if name in _OPTIM_EXPORTS:
        from . import optim
        return getattr(optim, name)
    if name in _TOPOLOGY_EXPORTS:
        from . import topology
        return getattr(topology, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_dir_name(dataset: str, seed: int, run_date: str = None) -> str:
    """返回稳定的 ``mnist_seed42`` 目录名；显式 run_date 仅用于读取旧目录。"""
    if run_date:
        return f"{dataset.lower()}_{run_date}_seed{seed}"
    return f"{dataset.lower()}_seed{seed}"


def get_result_dir(base_dir, dataset: str, category: str, sub: str = '',
                   seed: int = None, run_date: str = None) -> Path:
    """
    统一结果目录解析。

    有 seed 时（推荐）：
      base_dir / {dataset}_{date}_seed{seed} / {category} / {sub}

    无 seed 时（兼容旧调用）：
      base_dir / {dataset} / {category} / {sub}
    """
    if seed is not None:
        root = Path(base_dir) / run_dir_name(dataset, seed, run_date)
    else:
        root = Path(base_dir) / dataset.lower()
    p = root / category
    if sub:
        p = p / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    'check_dataset', 'check_all_datasets', 'download_dataset',
    'get_result_dir', 'run_dir_name',
    *_DATA_LOADER_EXPORTS, *_OPTIM_EXPORTS, *_TOPOLOGY_EXPORTS,
]
