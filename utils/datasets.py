"""Lightweight dataset status checks and lazy dataset downloads."""
from pathlib import Path


_DATASET_MARKERS = {
    'mnist': [
        'MNIST/raw/train-images-idx3-ubyte',
        'MNIST/raw/t10k-images-idx3-ubyte',
    ],
    'cifar10': [
        'cifar-10-batches-py/data_batch_1',
        'cifar-10-batches-py/test_batch',
    ],
}


def check_dataset(name: str, data_dir: str = './data') -> bool:
    """Return whether all marker files for one dataset exist."""
    root = Path(data_dir)
    markers = _DATASET_MARKERS.get(name.lower(), [])
    return bool(markers) and all((root / marker).exists() for marker in markers)


def check_all_datasets(data_dir: str = './data') -> dict:
    """Return dataset availability without importing the training stack."""
    return {name: check_dataset(name, data_dir) for name in _DATASET_MARKERS}


def download_dataset(name: str, data_dir: str = './data') -> None:
    """Download one dataset, importing torchvision only when requested."""
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dataset download requires torch and torchvision. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    name = name.lower()
    transform = transforms.ToTensor()
    if name == 'mnist':
        datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif name == 'cifar10':
        datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")
