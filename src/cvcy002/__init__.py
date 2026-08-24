# src/cvcy002/__init__.py

"""
cvcy002 - Clothes Segmentation Package
A deep learning pipeline for semantic segmentation of clothing items.
"""

__version__ = "0.1.0"
__author__ = "Abdulrahman Hassan"

# Optionally expose top-level modules for convenience
from . import data
from . import models
from . import training
from . import evaluation

__all__ = [
    "data",
    "models", 
    "training",
    "evaluation"
]