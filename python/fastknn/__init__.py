"""
FastKNN: High-performance K-Nearest Neighbors with multiple parallel backends
"""

from .knn import KNeighborsClassifier
from .backends import Backend

__version__ = "1.0.0"
__all__ = ["KNeighborsClassifier", "Backend"]
