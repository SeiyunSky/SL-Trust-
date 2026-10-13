"""Aggregators package"""
from .base import BaseAggregator
from .fedavg import FedAvgAggregator
from .krum import KrumAggregator
from .trimmed_mean import TrimmedMeanAggregator
from .balance import BALANCEAggregator
from .sama import SAMAAggregator
from .contra import CONTRAAggregator
from .contra_hard import CONTRAHardAggregator
from .sltrust import SLTrustAggregator

__all__ = [
    'BaseAggregator',
    'FedAvgAggregator',
    'KrumAggregator',
    'TrimmedMeanAggregator',
    'BALANCEAggregator',
    'SAMAAggregator',
    'CONTRAAggregator',
    'CONTRAHardAggregator',
    'SLTrustAggregator',
]
