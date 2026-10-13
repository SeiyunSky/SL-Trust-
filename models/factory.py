"""Model construction from experiment configuration."""
from . import SimpleCNN
from .resnet_cifar import CIFARResNet18


def build_model(config: dict, in_channels: int, num_classes: int):
    name = config.get('model', {}).get('name', 'simple_cnn').lower()
    if name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    if name == 'resnet18_cifar_bn':
        return CIFARResNet18(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"Unknown model: {name}")
