"""Aggregators package"""
from .base import BaseAggregator
from .fedavg import FedAvgAggregator
from .sama import SAMAAggregator
from .contra import CONTRAAggregator
from .contra_hard import CONTRAHardAggregator
from .sltrust import SLTrustAggregator

__all__ = [
    'BaseAggregator',
    'FedAvgAggregator',
    'SAMAAggregator',
    'CONTRAAggregator',
    'CONTRAHardAggregator',
    'SLTrustAggregator',
]
