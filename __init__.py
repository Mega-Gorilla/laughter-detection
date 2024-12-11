# laughter_detection/__init__.py
from .SimpleLaughterDetector import SimpleLaughterDetector
from .detector import LaughterDetector
from . import models
from . import configs

__all__ = ['SimpleLaughterDetector',
           'LaughterDetector',
           'models',
           'configs']