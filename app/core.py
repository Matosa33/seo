"""
Module core optimisé pour l'analyse sémantique des mots-clés.
Optimisé pour les performances et la gestion mémoire.
"""
from typing import List, Dict, Optional, Set
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
from dataclasses import dataclass
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from utils import Timer, safe_filename
import hashlib

@dataclass(frozen=True)
class KeywordData:
    """Structure de données immutable pour les informations de mots-clés"""
    keyword: str
    volume: int
    vector: Optional[np.ndarray] = None
    semantic_score: float = 0.0

class SmartCache:
    """Gestionnaire de cache intelligent avec limite de taille et LRU"""
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict = {}
        self._access_count: Dict = {}
        self._max_size = max_size
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[any]:
        with self._lock:
            if key in self._cache:
                self._access_count[key] += 1
                return self._cache[key]
            return None
            
    def set(self, key: str, value: any):
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Supprime 20% des entrées les moins utilisées
                items_to_remove = int(self._max_size * 0.2)
                sorted_items = sorted(
                    self._access_count.items(),
                    key=lambda x: x[1]
                )[:items_to_remove]
                
                for k, _ in sorted_items:
                    del self._cache[k]
                    del self._access_count[k]
                    
            self._cache[key] = value
            self._access_count[key] = 1
            
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            
    @property
    def size(self) -> int:
        return len(self._cache)

class OptimizedSemanticCore:
    """Core sémantique optimisé avec gestion efficace de la mémoire"""
    
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._vectors_cache_file = self.cache_dir / 'vectors_cache.npz'
        self._keywords_cache_file = self.cache_dir / 'keywords_cache.pkl'
        self._model_file = self.cache_dir / 'tfidf_model.pkl'
        
        # Initialisation des caches optimisés
        self.vector_cache = SmartCache(10000)
        self.similarity_cache = SmartCache(20000)
        
        # Initialisation unique du vectorizer
        self.tfidf = None
        self._load_or_init_model()
        
        # Données des mots-clés avec stockage optimisé
        self.keyword_data: Dict[str, KeywordData] = {}
        
        # Pool de threads pour les calculs parallèles
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def _load_or_init_model(self):
        """Charge ou initialise le modèle TF-IDF une seule fois"""
        if self._model_file.exists():
            self.tfidf = joblib.load(self._model_file)
        else:
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=5000,
                norm='l2'
            )
        
    def _vectorize_text(self, text: str) -> sp.csr_matrix:
        """Vectorisation optimisée avec gestion des formats sparse"""
        try:
            vector = self.tfidf.transform([text])
            if vector.nnz < vector.shape[1] * 0.1:  # Si moins de 10% non-nul
                return vector
            return vector.toarray()
        except Exception:
            return sp.csr_matrix((1, self.tfidf.get_feature_names_out().shape[0]))
            
    def load_keywords(self, data: pd.DataFrame) -> None:
        """Chargement optimisé avec cache persistant"""
        # Création d'une clé de hash pour les données
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(data).values
        ).hexdigest()
        
        # Vérification du cache
        cache_hash_file = self.cache_dir / 'data_hash.txt'
        if cache_hash_file.exists():
            with open(cache_hash_file, 'r') as f:
                cached_hash = f.read().strip()
                
            if cached_hash == data_hash:
                # Chargement depuis le cache
                self._load_from_cache()
                return

        # Si pas de cache valide, effectuer le traitement
        with Timer("Vectorisation"):
            if not hasattr(self.tfidf, 'vocabulary_') or not self.tfidf.vocabulary_:
                self.tfidf.fit(data['keyword'])
                joblib.dump(self.tfidf, self._model_file)
                
            vectors = self.tfidf.transform(data['keyword'])
            
        with Timer("Chargement"):
            # Traitement par lots pour réduire l'utilisation mémoire
            batch_size = 1000
            self.keyword_data = {}
            
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                batch_vectors = vectors[i:i+batch_size]
                
                for idx, row in batch.iterrows():
                    self.keyword_data[row['keyword']] = KeywordData(
                        keyword=row['keyword'],
                        volume=int(row['volume']),
                        vector=batch_vectors[idx-i].toarray()[0] if sp.issparse(batch_vectors[idx-i]) else batch_vectors[idx-i]
                    )
                    
            # Sauvegarde du cache
            self._save_to_cache(data_hash)

    def _save_to_cache(self, data_hash: str):
        """Sauvegarde optimisée du cache"""
        # Sauvegarde des vecteurs en format sparse
        vectors = sp.vstack([
            sp.csr_matrix(kd.vector) for kd in self.keyword_data.values()
        ])
        sp.save_npz(self._vectors_cache_file, vectors)
        
        # Sauvegarde des métadonnées
        metadata = {
            k: KeywordData(keyword=v.keyword, volume=v.volume, vector=None)
            for k, v in self.keyword_data.items()
        }
        joblib.dump(metadata, self._keywords_cache_file)
        
        # Sauvegarde du hash
        with open(self.cache_dir / 'data_hash.txt', 'w') as f:
            f.write(data_hash)
            
    def _load_from_cache(self):
        """Chargement optimisé depuis le cache"""
        vectors = sp.load_npz(self._vectors_cache_file)
        metadata = joblib.load(self._keywords_cache_file)
        
        self.keyword_data = {}
        for i, (keyword, data) in enumerate(metadata.items()):
            self.keyword_data[keyword] = KeywordData(
                keyword=data.keyword,
                volume=data.volume,
                vector=vectors[i].toarray()[0]
            )

    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calcul de similarité optimisé avec cache intelligent"""
        cache_key = f"{min(word1, word2)}|{max(word1, word2)}"
        
        # Vérification du cache
        cached = self.similarity_cache.get(cache_key)
        if cached is not None:
            return cached
            
        try:
            # Récupération des vecteurs
            vec1 = self.keyword_data[word1].vector
            vec2 = self.keyword_data[word2].vector
            
            if vec1 is None or vec2 is None:
                return 0.0
                
            # Protection contre les vecteurs nuls
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                # Calcul optimisé de la similarité cosinus
                similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
            
            # Mise en cache du résultat
            self.similarity_cache.set(cache_key, similarity)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"Erreur calcul similarité: {e}")  # Remplacé logger par print
            return 0.0
            
    def find_similar_keywords(self,
                            keyword: str,
                            min_similarity: float = 0.3,
                            max_results: int = 10,
                            min_volume: int = 0) -> List[Dict]:
        """Recherche optimisée de mots-clés similaires"""
        if keyword not in self.keyword_data:
            return []
            
        # Calcul parallèle des similarités
        with ThreadPoolExecutor() as executor:
            futures = []
            for kw in self.keyword_data:
                if kw != keyword and self.keyword_data[kw].volume >= min_volume:
                    futures.append(
                        executor.submit(self.calculate_similarity, keyword, kw)
                    )
                    
            # Collecte des résultats
            results = []
            for future, kw in zip(futures, [k for k in self.keyword_data if k != keyword]):
                try:
                    similarity = future.result()
                    if similarity >= min_similarity:
                        results.append({
                            'keyword': kw,
                            'similarity': round(similarity, 3),
                            'volume': self.keyword_data[kw].volume
                        })
                except Exception as e:
                    self.logger.error(f"Erreur calcul similarité pour {kw}: {e}")
                    
        return sorted(
            results,
            key=lambda x: (-x['similarity'], -x['volume'])
        )[:max_results]
        
    def suggest_keywords(self,
                    current_keywords: Set[str],
                    max_suggestions: int = 10,
                    min_similarity: float = 0.3,
                    min_volume: int = 0) -> List[Dict]:
        """Suggestions sémantiques avancées pour cocons SEO"""
        if not current_keywords:
            return []

        valid_keywords = {k for k in current_keywords if k in self.keyword_data}
        if not valid_keywords:
            return []

        candidates = {}
        max_volume = max(data.volume for data in self.keyword_data.values())
        
        # Calcul du centroïde sémantique du cocon
        cocon_vectors = [self.keyword_data[kw].vector for kw in valid_keywords]
        cocon_centroid = np.mean(cocon_vectors, axis=0)
        
        # Calcul de la dispersion sémantique du cocon
        semantic_dispersion = np.std([
            self.calculate_vector_similarity(vec, cocon_centroid)
            for vec in cocon_vectors
        ])

        for candidate, data in self.keyword_data.items():
            if candidate in valid_keywords or data.volume < min_volume:
                continue

            # Calcul des similarités avec structure hiérarchique
            hierarchical_similarities = {
                'direct': [],      # Similarités directes avec les mots-clés
                'secondary': [],   # Similarités avec les mots connexes
                'centroid': 0.0    # Similarité avec le centroïde du cocon
            }

            # 1. Similarités directes
            for current_kw in valid_keywords:
                sim = self.calculate_similarity(candidate, current_kw)
                if sim >= min_similarity:
                    hierarchical_similarities['direct'].append(sim)

            if not hierarchical_similarities['direct']:
                continue

            # 2. Similarités secondaires (avec les mots connexes du cocon)
            connected_keywords = set()
            for kw in valid_keywords:
                if kw in self.keyword_data:
                    related = self.get_related_keywords(kw, min_similarity)
                    connected_keywords.update(related)
            
            for related_kw in connected_keywords:
                if related_kw not in valid_keywords:
                    sim = self.calculate_similarity(candidate, related_kw)
                    if sim >= min_similarity:
                        hierarchical_similarities['secondary'].append(sim)

            # 3. Similarité avec le centroïde
            hierarchical_similarities['centroid'] = self.calculate_vector_similarity(
                data.vector, 
                cocon_centroid
            )

            # Calculs des scores avancés
            direct_score = np.mean(hierarchical_similarities['direct'])
            secondary_score = (np.mean(hierarchical_similarities['secondary']) 
                             if hierarchical_similarities['secondary'] else 0)
            centroid_score = hierarchical_similarities['centroid']
            
            # Normalisation du volume avec échelle logarithmique
            volume_score = np.log1p(data.volume) / np.log1p(max_volume)
            
            # Score sémantique pondéré
            semantic_score = (
                0.50 * direct_score +          # Similarité directe
                0.25 * secondary_score +       # Similarité secondaire
                0.25 * centroid_score          # Cohérence avec le cocon
            )

            # Score de positionnement dans la hiérarchie
            hierarchy_score = (
                0.6 * max(hierarchical_similarities['direct']) +  # Meilleure connexion
                0.4 * (1 - semantic_dispersion)                  # Cohésion du cocon
            )

            # Pertinence finale multi-facteurs
            relevance = (
                0.40 * semantic_score +     # Force sémantique
                0.25 * hierarchy_score +    # Position hiérarchique
                0.20 * volume_score +       # Importance SEO
                0.15 * (1 - (len(hierarchical_similarities['direct']) / len(valid_keywords)))  # Originalité
            )

            candidates[candidate] = {
                'keyword': candidate,
                'similarity': round(semantic_score, 3),
                'volume': data.volume,
                'relevance': round(relevance, 3),
                'hierarchy_score': round(hierarchy_score, 3),
                'direct_connections': len(hierarchical_similarities['direct']),
                'secondary_connections': len(hierarchical_similarities['secondary']),
                'centroid_similarity': round(centroid_score, 3)
            }

        # Tri intelligent avec diversification
        suggestions = self._diversify_suggestions(
            candidates.values(),
            max_suggestions,
            semantic_dispersion
        )

        return suggestions

    def _diversify_suggestions(self, candidates: List[Dict], 
                             max_count: int, 
                             dispersion: float) -> List[Dict]:
        """Diversification intelligente des suggestions"""
        if not candidates:
            return []

        selected = []
        remaining = list(candidates)
        
        # Sélection du premier candidat (meilleure pertinence)
        remaining.sort(key=lambda x: (-x['relevance'], -x['volume']))
        selected.append(remaining.pop(0))
        
        # Sélection itérative avec diversification
        while len(selected) < max_count and remaining:
            scores = []
            for candidate in remaining:
                # Score de diversité par rapport aux sélectionnés
                diversity_score = np.mean([
                    1 - self.calculate_similarity(candidate['keyword'], s['keyword'])
                    for s in selected
                ])
                
                # Score combiné (pertinence + diversité)
                combined_score = (
                    0.7 * candidate['relevance'] +
                    0.3 * diversity_score
                )
                scores.append((candidate, combined_score))
            
            # Sélection du meilleur candidat
            best_candidate = max(scores, key=lambda x: x[1])[0]
            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected
        
    def analyze_keyword_group(self,
                            keywords: Set[str],
                            min_similarity: float = 0.3) -> Dict[str, Dict]:
        """Analyse optimisée d'un groupe de mots-clés"""
        valid_keywords = {k for k in keywords if k in self.keyword_data}
        if not valid_keywords:
            return {}
            
        analysis = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                kw: [
                    executor.submit(self.calculate_similarity, kw, other_kw)
                    for other_kw in valid_keywords if other_kw != kw
                ]
                for kw in valid_keywords
            }
            
            for kw, kw_futures in futures.items():
                try:
                    similarities = [f.result() for f in kw_futures]
                    related = [
                        {
                            'keyword': other_kw,
                            'similarity': sim,
                            'volume': self.keyword_data[other_kw].volume
                        }
                        for other_kw, sim in zip(
                            [k for k in valid_keywords if k != kw],
                            similarities
                        )
                        if sim >= min_similarity
                    ]
                    
                    analysis[kw] = {
                        'volume': self.keyword_data[kw].volume,
                        'semantic_strength': round(np.mean(similarities), 3),
                        'related_keywords': sorted(
                            related,
                            key=lambda x: -x['similarity']
                        )
                    }
                except Exception as e:
                    self.logger.error(f"Erreur analyse pour {kw}: {e}")
                    
        return analysis
        
    def export_data(self, keywords: Set[str]) -> Dict:
        """Export optimisé des données"""
        analysis = self.analyze_keyword_group(keywords)
        
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_keywords': len(keywords),
                'avg_volume': round(
                    np.mean([
                        self.keyword_data[kw].volume 
                        for kw in keywords 
                        if kw in self.keyword_data
                    ]),
                    2
                )
            },
            'keywords': [
                {
                    'keyword': kw,
                    **analysis.get(kw, {
                        'volume': self.keyword_data[kw].volume,
                        'semantic_strength': 0.0,
                        'related_keywords': []
                    })
                }
                for kw in keywords
                if kw in self.keyword_data
            ]
        }
        
        # Sauvegarde
        export_file = self.cache_dir / f'export_{safe_filename("_".join(sorted(keywords)))}.json'
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        return export_data
        
    def cleanup(self) -> Dict[str, int]:
        """Nettoyage complet des caches"""
        stats = {
            'vector_cache_cleared': self.vector_cache.size,
            'similarity_cache_cleared': self.similarity_cache.size
        }
        
        self.vector_cache.clear()
        self.similarity_cache.clear()
        
        return stats
        
    def get_stats(self) -> Dict[str, any]:
        """Statistiques d'utilisation optimisées"""
        try:
            return {
                'total_keywords': len(self.keyword_data),
                'vector_cache_size': self.vector_cache.size,
                'similarity_cache_size': self.similarity_cache.size,
                'total_volume': sum(
                    data.volume for data in self.keyword_data.values()
                ),
                'memory_usage_mb': sum(
                    sys.getsizeof(data) for data in self.keyword_data.values()
                ) / (1024 * 1024)
            }
        except Exception as e:
            self.loggerself.logger.error(f"Erreur calcul stats: {e}")
            return {
                'total_keywords': 0,
                'vector_cache_size': 0,
                'similarity_cache_size': 0,
                'total_volume': 0,
                'memory_usage_mb': 0
            }
            
    def get_keyword_info(self, keyword: str) -> Dict:
        """Récupération optimisée des informations d'un mot-clé"""
        if keyword not in self.keyword_data:
            return {}
            
        data = self.keyword_data[keyword]
        
        return {
            'keyword': keyword,
            'volume': data.volume,
            'semantic_score': data.semantic_score,
            'vector_size': (
                sys.getsizeof(data.vector) 
                if data.vector is not None 
                else 0
            ) / 1024,  # KB
            'is_cached': self.vector_cache.get(keyword) is not None
        }

    def batch_process_similarities(self,
                                 keywords: List[str],
                                 threshold: float = 0.3) -> np.ndarray:
        """Calcul optimisé des similarités par lot"""
        n = len(keywords)
        similarities = np.zeros((n, n))
        
        # Création de la matrice de vecteurs
        vectors = np.vstack([
            self.keyword_data[kw].vector 
            for kw in keywords
        ])
        
        # Calcul matriciel des similarités
        norms = np.linalg.norm(vectors, axis=1)
        similarities = np.dot(vectors, vectors.T) / np.outer(norms, norms)
        
        # Application du seuil
        similarities[similarities < threshold] = 0
        
        return similarities
        
    def precompute_common_keywords(self, top_n: int = 1000):
        """Pré-calcul des similarités pour les mots-clés fréquents"""
        if not self.keyword_data:
            return
            
        common_keywords = sorted(
            self.keyword_data.items(),
            key=lambda x: x[1].volume,
            reverse=True
        )[:top_n]
        
        self.logger.info(f"Pré-calcul pour {len(common_keywords)} mots-clés...")
        
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.calculate_similarity,
                    kw1[0], kw2[0]
                )
                for i, kw1 in enumerate(common_keywords)
                for kw2 in common_keywords[i+1:]
            ]
            
            # Collecte silencieuse des résultats
            for future in futures:
                try:
                    future.result()
                except Exception:
                    pass
                    
        self.logger.info("Pré-calcul terminé")

    @staticmethod
    def calculate_vector_similarity(vec1: np.ndarray, 
                                  vec2: np.ndarray) -> float:
        """Calcul optimisé de similarité entre vecteurs"""
        try:
            if sp.issparse(vec1):
                vec1 = vec1.toarray().flatten()
            if sp.issparse(vec2):
                vec2 = vec2.toarray().flatten()
                
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
            
        except Exception:
            return 0.0

    def optimize_memory(self, max_memory_mb: float = 1000.0):
        """Optimisation de l'utilisation mémoire"""
        current_memory = self.get_stats()['memory_usage_mb']
        
        if current_memory > max_memory_mb:
            self.logger.info(f"Optimisation mémoire: {current_memory:.1f}MB -> {max_memory_mb}MB")
            
            # 1. Nettoyage des caches
            self.cleanup()
            
            # 2. Conversion des vecteurs en format sparse si possible
            for kw, data in self.keyword_data.items():
                if data.vector is not None:
                    vec = data.vector
                    if not sp.issparse(vec):
                        sparsity = np.count_nonzero(vec) / vec.size
                        if sparsity < 0.1:  # Si moins de 10% non-nul
                            self.keyword_data[kw] = KeywordData(
                                keyword=data.keyword,
                                volume=data.volume,
                                vector=sp.csr_matrix(vec),
                                semantic_score=data.semantic_score
                            )
            
            # 3. Force le garbage collector
            import gc
            gc.collect()
            
            new_memory = self.get_stats()['memory_usage_mb']
            self.logger.info(f"Mémoire après optimisation: {new_memory:.1f}MB")
            
    def __enter__(self):
        """Support du context manager"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage à la sortie du context manager"""
        self.cleanup()
        self._executor.shutdown()
        
# Tests unitaires et exemples d'utilisation
if __name__ == "__main__":
    def run_benchmark():
        """Test de performance"""
        print("\n🔬 Démarrage du benchmark...")
        
        # Génération de données de test
        import numpy as np
        test_size = 1000
        
        keywords = [f"keyword_{i}" for i in range(test_size)]
        volumes = np.random.randint(100, 10000, size=test_size)
        
        test_data = pd.DataFrame({
            'keyword': keywords,
            'volume': volumes
        })
        
        with Timer("Test complet"), OptimizedSemanticCore() as core:
            # Test chargement
            with Timer("Chargement"):
                core.load_keywords(test_data)
                
            # Test similarités
            with Timer("Calcul similarités"):
                for _ in range(100):
                    kw1, kw2 = np.random.choice(keywords, 2)
                    core.calculate_similarity(kw1, kw2)
                    
            # Test suggestions
            with Timer("Suggestions"):
                suggestions = core.suggest_keywords(
                    set(np.random.choice(keywords, 5)),
                    max_suggestions=10
                )
                
            # Test analyse
            with Timer("Analyse"):
                analysis = core.analyze_keyword_group(
                    set(np.random.choice(keywords, 10))
                )
                
            # Stats finales
            stats = core.get_stats()
            print("\n📊 Statistiques finales:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"- {key}: {value:.2f}")
                else:
                    print(f"- {key}: {value}")
                    
    def run_basic_tests():
        """Tests basiques de fonctionnalité"""
        print("\n🧪 Tests basiques...")
        
        test_data = pd.DataFrame({
            'keyword': ['marketing digital', 'seo', 'référencement'],
            'volume': [1000, 800, 600]
        })
        
        with OptimizedSemanticCore() as core:
            # Test chargement
            core.load_keywords(test_data)
            assert len(core.keyword_data) == 3
            
            # Test similarité
            sim = core.calculate_similarity('marketing digital', 'seo')
            assert 0 <= sim <= 1
            
            # Test suggestions
            suggs = core.suggest_keywords({'marketing digital'})
            assert len(suggs) > 0
            
            print("✅ Tests basiques OK")
            
    def run_stress_test():
        """Test de charge"""
        print("\n💪 Test de charge...")
        
        # Génération de données
        test_size = 5000
        keywords = [f"keyword_{i}" for i in range(test_size)]
        volumes = np.random.randint(100, 10000, size=test_size)
        
        test_data = pd.DataFrame({
            'keyword': keywords,
            'volume': volumes
        })
        
        with Timer("Test de charge"), OptimizedSemanticCore() as core:
            core.load_keywords(test_data)
            
            # Test calculs intensifs
            with ThreadPoolExecutor() as executor:
                futures = []
                for _ in range(1000):
                    kw1, kw2 = np.random.choice(keywords, 2)
                    futures.append(
                        executor.submit(core.calculate_similarity, kw1, kw2)
                    )
                    
                for future in futures:
                    future.result()
                    
            print("✅ Test de charge OK")
            
    try:
        print("🚀 Démarrage des tests...")
        run_basic_tests()
        run_benchmark()
        run_stress_test()
        print("\n✨ Tous les tests sont passés avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur pendant les tests: {str(e)}")
        raise