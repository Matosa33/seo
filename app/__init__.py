"""
Package pour l'application de génération de cocons sémantiques
"""
from .core import SemanticCore
from .app import SemanticCocoonApp
from .utils import calculate_similarity, memoize

__all__ = [
    'SemanticCore',
    'SemanticCocoonApp',
    'calculate_similarity',
    'memoize'
]

__version__ = '1.0.0'