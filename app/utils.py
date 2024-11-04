# utils.py
"""
Module utilitaire pour l'application de cocons sémantiques.
Contient les fonctions de cache et les outils communs.
"""
from typing import Callable, Dict, Any, Optional
from functools import wraps
import numpy as np
from numpy.linalg import norm
from numpy import dot
import threading
import logging
from pathlib import Path
import json
from datetime import datetime
import hashlib

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheManager:
    """Gestionnaire de cache thread-safe avec persistance"""
    
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        with self._lock:
            return self._cache.get(key)
            
    def set(self, key: str, value: Any) -> None:
        """Définit une valeur dans le cache"""
        with self._lock:
            self._cache[key] = value
            
    def clear(self) -> None:
        """Vide le cache"""
        with self._lock:
            self._cache.clear()
            
    def save_to_disk(self, filename: str) -> None:
        """Sauvegarde le cache sur le disque"""
        cache_path = self.cache_dir / filename
        with self._lock:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'cache': {
                            k: v for k, v in self._cache.items()
                            if isinstance(v, (dict, list, str, int, float))
                        }
                    }, f, ensure_ascii=False, indent=2)
                logger.info(f"Cache sauvegardé dans {cache_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du cache: {e}")
                
    def load_from_disk(self, filename: str) -> None:
        """Charge le cache depuis le disque"""
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    with self._lock:
                        self._cache.update(data.get('cache', {}))
                logger.info(f"Cache chargé depuis {cache_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du cache: {e}")

def memoize(func: Callable):
    """
    Décorateur de mise en cache des résultats de fonction.
    Utilise un hash des arguments comme clé de cache.
    
    Args:
        func: La fonction à mettre en cache
        
    Returns:
        La fonction décorée avec gestion du cache
    """
    cache = {}
    lock = threading.Lock()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Création d'une clé de cache unique
        key = hashlib.md5(
            str((args, sorted(kwargs.items()))).encode()
        ).hexdigest()
        
        with lock:
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
    
    def cache_info() -> Dict[str, int]:
        """Retourne des informations sur l'état du cache"""
        return {
            'size': len(cache),
            'bytes': sum(
                len(str(v).encode()) 
                for v in cache.values()
            )
        }
    
    def cache_clear() -> None:
        """Vide le cache"""
        with lock:
            cache.clear()
    
    # Ajout des méthodes utilitaires
    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper

@memoize
def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    
    Args:
        vec1: Premier vecteur numpy
        vec2: Second vecteur numpy
        
    Returns:
        float: Score de similarité entre 0 et 1
    """
    try:
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Vérification des dimensions
        if vec1.shape != vec2.shape:
            logger.warning(
                f"Dimensions incompatibles: {vec1.shape} != {vec2.shape}"
            )
            return 0.0
            
        # Calcul de la similarité cosinus
        similarity = float(
            dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        )
        
        # Normalisation pour éviter les erreurs numériques
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul de similarité: {e}")
        return 0.0

class Timer:
    """Classe utilitaire pour mesurer le temps d'exécution"""
    
    def __init__(self, name: str = None):
        self.name = name or 'Timer'
        
    def __enter__(self):
        self.start = datetime.now()
        return self
        
    def __exit__(self, *args):
        self.end = datetime.now()
        self.duration = self.end - self.start
        logger.info(
            f"{self.name}: {self.duration.total_seconds():.3f}s"
        )

def safe_filename(text: str) -> str:
    """
    Convertit un texte en nom de fichier sûr.
    
    Args:
        text: Le texte à convertir
        
    Returns:
        str: Le nom de fichier sécurisé
    """
    # Remplace les caractères non sûrs
    safe = "".join(
        c if c.isalnum() or c in ('-', '_') else '_'
        for c in text.lower()
    )
    return safe[:100]  # Limite la longueur

def format_number(n: int) -> str:
    """
    Formate un nombre pour l'affichage.
    
    Args:
        n: Le nombre à formater
        
    Returns:
        str: Le nombre formaté
    """
    if n < 1000:
        return str(n)
    elif n < 1000000:
        return f"{n/1000:.1f}k"
    else:
        return f"{n/1000000:.1f}M"

# Export des éléments publics
__all__ = [
    'memoize',
    'calculate_similarity',
    'CacheManager',
    'Timer',
    'safe_filename',
    'format_number'
]