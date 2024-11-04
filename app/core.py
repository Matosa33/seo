"""
Module core pour l'analyse sémantique des mots-clés.
"""
import numpy as np
from typing import List, Dict, Optional, Set
import pandas as pd
from pathlib import Path
import joblib
import logging
from datetime import datetime
from dataclasses import dataclass
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
from numpy import dot

# Import des utilitaires
from utils import calculate_similarity, CacheManager, Timer, safe_filename

@dataclass
class KeywordData:
    """Structure de données pour les informations de mots-clés"""
    keyword: str
    volume: int
    difficulty: float = 0.0
    vector: Optional[np.ndarray] = None
    cluster_id: int = -1
    semantic_score: float = 0.0

class LightweightSemanticCore:
    """Version optimisée pour ordinateurs portables"""
    
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Configuration du logging
        self.logger = logging.getLogger(__name__)
        
        # Initialisation des caches
        self.cache_manager = CacheManager(cache_dir)
        self.vector_cache = {}
        self.similarity_cache = {}
        self._cache_lock = threading.Lock()
        
        # Données des mots-clés
        self.keyword_data: Dict[str, KeywordData] = {}
        
        # Initialisation des modèles légers
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialise les modèles légers"""
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=10000  # Limite pour la mémoire
        )
        
        # Chargement du cache des vecteurs s'il existe
        cache_file = self.cache_dir / 'vectors.joblib'
        if cache_file.exists():
            try:
                self.vector_cache = joblib.load(cache_file)
                self.logger.info(f"Cache vectoriel chargé: {len(self.vector_cache)} entrées")
            except Exception as e:
                self.logger.error(f"Erreur chargement cache: {e}")
                self.vector_cache = {}
            
    def _compute_ngram_vector(self, text: str) -> np.ndarray:
        """Calcule un vecteur basé sur les n-grammes"""
        # Création d'une matrice TF-IDF pour un seul texte
        try:
            vector = self.tfidf.transform([text])
            return vector.toarray()[0]
        except Exception:
            # Si le mot n'est pas dans le vocabulaire
            return np.zeros(self.tfidf.max_features)
            
    def _preprocess_keywords(self, keywords: List[str]):
        """Pré-traitement efficace des mots-clés"""
        self.logger.info("Démarrage du pré-traitement des mots-clés...")
        
        # Mise à jour du vocabulaire TF-IDF
        self.tfidf.fit(keywords)
        
        # Pré-calcul des vecteurs en batch
        with Timer("Calcul des vecteurs"):
            vectors = self.tfidf.transform(keywords)
        
        # Mise en cache des vecteurs
        with Timer("Mise en cache"):
            for idx, keyword in enumerate(keywords):
                self.vector_cache[keyword] = vectors[idx].toarray()[0]
            
        # Sauvegarde du cache
        with Timer("Sauvegarde du cache"):
            joblib.dump(self.vector_cache, self.cache_dir / 'vectors.joblib')
            
        self.logger.info(f"Pré-traitement terminé pour {len(keywords)} mots-clés")

    def load_keywords(self, 
                     data: pd.DataFrame, 
                     keyword_col: str = 'keyword',
                     volume_col: str = 'volume',
                     difficulty_col: Optional[str] = None) -> None:
        """
        Charge les données de mots-clés depuis un DataFrame
        """
        if not all(col in data.columns for col in [keyword_col, volume_col]):
            raise ValueError(f"Colonnes requises manquantes: {keyword_col}, {volume_col}")
        
        # Nettoyage des données
        data[keyword_col] = data[keyword_col].str.lower().str.strip()
        data = data.drop_duplicates(subset=[keyword_col])
        
        # Chargement des données
        with Timer("Chargement des mots-clés"):
            for _, row in tqdm(data.iterrows(), total=len(data), desc="Chargement"):
                keyword = row[keyword_col]
                volume = int(row[volume_col])
                difficulty = float(row.get(difficulty_col, 0.0)) if difficulty_col else 0.0
                
                self.keyword_data[keyword] = KeywordData(
                    keyword=keyword,
                    volume=volume,
                    difficulty=difficulty
                )
        
        # Pré-traitement des vecteurs
        self._preprocess_keywords(list(self.keyword_data.keys()))
        
        self.logger.info(f"Chargement terminé: {len(self.keyword_data)} mots-clés")
        
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calcul de similarité optimisé"""
        # Vérification du cache de similarité
        cache_key = f"{word1}|{word2}"
        with self._cache_lock:
            if cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]
        
        try:
            # Récupération ou calcul des vecteurs
            vec1 = self.vector_cache.get(word1)
            if vec1 is None:
                vec1 = self._compute_ngram_vector(word1)
                
            vec2 = self.vector_cache.get(word2)
            if vec2 is None:
                vec2 = self._compute_ngram_vector(word2)
            
            if vec1 is None or vec2 is None:
                return 0.0
            
            # Calcul de la similarité cosinus
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
            
            # Mise en cache
            with self._cache_lock:
                self.similarity_cache[cache_key] = similarity
                
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Erreur de calcul de similarité: {e}")
            return 0.0

    def find_similar_keywords(self, 
                            keyword: str,
                            min_similarity: float = 0.5,
                            max_results: int = 10,
                            min_volume: int = 0) -> List[Dict]:
        """Trouve les mots-clés similaires avec filtrage"""
        if keyword not in self.keyword_data:
            return []
            
        results = []
        
        # Calcul par lots pour l'efficacité
        batch_size = 1000
        all_keywords = list(self.keyword_data.keys())
        
        for i in range(0, len(all_keywords), batch_size):
            batch = all_keywords[i:i+batch_size]
            
            for kw in batch:
                if kw != keyword and self.keyword_data[kw].volume >= min_volume:
                    similarity = self.calculate_similarity(keyword, kw)
                    if similarity >= min_similarity:
                        results.append({
                            'keyword': kw,
                            'similarity': round(similarity, 3),
                            'volume': self.keyword_data[kw].volume
                        })
        
        return sorted(
            results,
            key=lambda x: (x['similarity'], x['volume']),
            reverse=True
        )[:max_results]

    def suggest_keywords(self,
                        current_keywords: Set[str],
                        max_suggestions: int = 10,
                        min_similarity: float = 0.3,
                        min_volume: int = 0,
                        min_distance: float = 0.2) -> List[Dict]:
        """
        Suggestions optimisées avec contrôle de distance
        """
        # Validation des mots-clés existants
        valid_current_keywords = {
            kw for kw in current_keywords 
            if kw in self.keyword_data
        }
        
        if not valid_current_keywords:
            self.logger.warning("Aucun mot-clé valide dans les mots-clés actuels")
            return []

        suggestions = []
        seen_keywords = set(valid_current_keywords)
        
        # Parcours optimisé du dictionnaire de mots-clés
        all_keywords = list(self.keyword_data.keys())
        batch_size = 1000
        
        for i in range(0, len(all_keywords), batch_size):
            batch = all_keywords[i:i+batch_size]
            
            for kw in batch:
                # Vérifications préliminaires
                if (kw in seen_keywords or 
                    self.keyword_data[kw].volume < min_volume):
                    continue
                
                try:
                    # Calcul des similarités avec gestion d'erreurs
                    similarities = []
                    for current_kw in valid_current_keywords:
                        try:
                            sim = self.calculate_similarity(kw, current_kw)
                            if sim is not None:
                                similarities.append(sim)
                        except Exception as e:
                            self.logger.error(f"Erreur calcul similarité {kw}/{current_kw}: {e}")
                            continue
                    
                    if not similarities:
                        continue
                        
                    avg_similarity = np.mean(similarities)
                    
                    # Vérification de la distance minimale
                    too_close = False
                    if min_distance > 0:
                        for existing in suggestions:
                            try:
                                distance = 1 - self.calculate_similarity(
                                    kw, 
                                    existing['keyword']
                                )
                                if distance < min_distance:
                                    too_close = True
                                    break
                            except Exception as e:
                                self.logger.error(f"Erreur calcul distance {kw}: {e}")
                                continue
                    
                    if too_close:
                        continue
                    
                    # Ajout de la suggestion si elle correspond aux critères
                    if avg_similarity >= min_similarity:
                        suggestion = {
                            'keyword': kw,
                            'similarity': round(avg_similarity, 3),
                            'volume': self.keyword_data[kw].volume,
                            'relevance': round(
                                avg_similarity * (1 + np.log1p(
                                    self.keyword_data[kw].volume / 1000
                                )), 
                                3
                            ),
                            'matches': len([
                                s for s in similarities 
                                if s >= min_similarity
                            ]),
                            'intent_match': round(
                                len(set(kw.split()) & 
                                    set(' '.join(valid_current_keywords).split())) / 
                                len(set(kw.split())), 
                                3
                            )
                        }
                        suggestions.append(suggestion)
                        
                        # Optimisation: arrêt si max_suggestions atteint
                        if len(suggestions) >= max_suggestions * 2:
                            break
                            
                except Exception as e:
                    self.logger.error(f"Erreur traitement mot-clé {kw}: {e}")
                    continue
        
        # Tri final et limitation du nombre de suggestions
        try:
            suggestions.sort(key=lambda x: (-x['relevance'], -x['volume']))
            return suggestions[:max_suggestions]
        except Exception as e:
            self.logger.error(f"Erreur tri final des suggestions: {e}")
            return suggestions[:max_suggestions]  # Retour non trié en cas d'erreur
        
    def cluster_suggestions(self, suggestions: List[Dict], threshold: float = 0.5) -> Dict[int, List[str]]:
        """
        Regroupe les suggestions en clusters basés sur leur similarité sémantique.
        
        Args:
            suggestions: Liste des suggestions (dictionnaires avec clé 'keyword')
            threshold: Seuil de similarité pour le clustering (0.0 à 1.0)
            
        Returns:
            Dict[int, List[str]]: Dictionnaire des clusters {id_cluster: [mots-clés]}
        """
        if not suggestions:
            return {}
            
        # Extraction des mots-clés
        keywords = [s['keyword'] for s in suggestions]
        
        # Initialisation des clusters
        clusters = {}
        assigned_keywords = set()
        cluster_id = 0
        
        try:
            # Pour chaque mot-clé non assigné
            for keyword in keywords:
                if keyword in assigned_keywords:
                    continue
                    
                # Création d'un nouveau cluster
                current_cluster = [keyword]
                assigned_keywords.add(keyword)
                
                # Recherche des mots-clés similaires
                for other_keyword in keywords:
                    if other_keyword != keyword and other_keyword not in assigned_keywords:
                        similarity = self.calculate_similarity(keyword, other_keyword)
                        if similarity >= threshold:
                            current_cluster.append(other_keyword)
                            assigned_keywords.add(other_keyword)
                
                # Sauvegarde du cluster s'il contient plus d'un mot-clé
                if len(current_cluster) > 1:
                    clusters[cluster_id] = current_cluster
                    cluster_id += 1
                    
            # Ajout des mots-clés isolés dans un cluster spécial
            remaining = [k for k in keywords if k not in assigned_keywords]
            if remaining:
                clusters[cluster_id] = remaining
                
            return clusters
            
        except Exception as e:
            self.logger.error(f"Erreur lors du clustering des suggestions: {e}")
            return {0: keywords}  # En cas d'erreur, retourne tous les mots-clés dans un seul cluster

    def analyze_keyword_group(self,
                            keywords: Set[str],
                            min_similarity: float = 0.5) -> Dict[str, Dict]:
        """Analyse un groupe de mots-clés"""
        cache_key = f"analysis_{'-'.join(sorted(keywords))}"
        cached = self.cache_manager.get(cache_key)
        
        if cached:
            return cached
            
        analysis = {}
        for keyword in keywords:
            if keyword not in self.keyword_data:
                continue
                
            related = []
            similarities = []
            
            for other_kw in keywords:
                if other_kw != keyword:
                    similarity = self.calculate_similarity(keyword, other_kw)
                    if similarity >= min_similarity:
                        related.append({
                            'keyword': other_kw,
                            'similarity': round(similarity, 3),
                            'volume': self.keyword_data[other_kw].volume
                        })
                        similarities.append(similarity)
            
            analysis[keyword] = {
                'volume': self.keyword_data[keyword].volume,
                'semantic_strength': round(np.mean(similarities), 3) if similarities else 0,
                'related_keywords': sorted(
                    related,
                    key=lambda x: x['similarity'],
                    reverse=True
                ),
                'cluster_score': len(related)
            }
        
        self.cache_manager.set(cache_key, analysis)
        return analysis

    def export_cocoon(self, keywords: Set[str]) -> Dict:
        """Exporte les données du cocon sémantique"""
        with Timer("Export du cocon"):
            analysis = self.analyze_keyword_group(keywords)
            clusters = self._generate_clusters(keywords, analysis)
            
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_keywords': len(keywords),
                    'avg_volume': round(
                        np.mean([self.keyword_data[kw].volume for kw in keywords]),
                        2
                    )
                },
                'keywords': [
                    {
                        'keyword': kw,
                        **analysis[kw]
                    }
                    for kw in keywords
                ],
                'clusters': clusters
            }
            
            # Sauvegarde du cocon
            export_file = self.cache_dir / f'cocon_{safe_filename("_".join(sorted(keywords)))}.json'
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
                
            return export_data

    def _generate_clusters(self,
                         keywords: Set[str],
                         analysis: Dict) -> List[Dict]:
        """Génère des clusters pour le cocon"""
        used_keywords = set()
        clusters = []
        
        # Tri par force sémantique
        sorted_keywords = sorted(
            keywords,
            key=lambda k: analysis[k]['semantic_strength'],
            reverse=True
        )
        
        for keyword in sorted_keywords:
            if keyword in used_keywords:
                continue
                
            cluster = {
                'center': keyword,
                'keywords': [keyword],
                'total_volume': analysis[keyword]['volume'],
                'avg_similarity': analysis[keyword]['semantic_strength']
            }
            
            used_keywords.add(keyword)
            
            # Ajout des mots-clés fortement liés
            for related in analysis[keyword]['related_keywords']:
                if related['keyword'] not in used_keywords:
                    cluster['keywords'].append(related['keyword'])
                    cluster['total_volume'] += related['volume']
                    used_keywords.add(related['keyword'])
            
            if len(cluster['keywords']) > 1:
                clusters.append(cluster)
        
        return clusters

    def clear_caches(self) -> None:
        """Nettoie tous les caches"""
        self.vector_cache.clear()
        self.similarity_cache.clear()
        self.cache_manager.clear()

    def get_stats(self) -> Dict[str, any]:
        """Retourne les statistiques du core"""
        return {
            'total_keywords': len(self.keyword_data),
            'cached_vectors': len(self.vector_cache),
            'total_volume': sum(data.volume for data in self.keyword_data.values()),
            'avg_difficulty': np.mean([
                data.difficulty 
                for data in self.keyword_data.values()
            ]),
            'memory_usage': {
                'vector_cache_mb': sum(
                    sys.getsizeof(vector) 
                    for vector in self.vector_cache.values()
                ) / (1024 * 1024),
                'similarity_cache_mb': sys.getsizeof(self.similarity_cache) / (1024 * 1024),
                'total_keywords_mb': sum(
                    sys.getsizeof(data) 
                    for data in self.keyword_data.values()
                ) / (1024 * 1024)
            },
            'cache_info': {
                'vectors': len(self.vector_cache),
                'similarities': len(self.similarity_cache)
            }
        }

    def get_keyword_stats(self, keyword: str) -> Dict:
        """Récupère les statistiques détaillées d'un mot-clé"""
        if keyword not in self.keyword_data:
            return {}
            
        data = self.keyword_data[keyword]
        vector = self.vector_cache.get(keyword)
        
        return {
            'keyword': keyword,
            'volume': data.volume,
            'difficulty': data.difficulty,
            'has_vector': vector is not None,
            'vector_dim': len(vector) if vector is not None else 0,
            'cached': keyword in self.vector_cache,
            'semantic_score': data.semantic_score,
            'cluster_id': data.cluster_id,
            'memory_usage': {
                'vector_bytes': sys.getsizeof(vector) if vector is not None else 0,
                'data_bytes': sys.getsizeof(data)
            }
        }

    def cleanup_caches(self, max_cache_size_mb: float = 1000) -> dict:
        """
        Nettoie les caches si leur taille dépasse la limite
        
        Args:
            max_cache_size_mb (float): Taille maximale du cache en MB
            
        Returns:
            dict: Statistiques du nettoyage
        """
        stats_before = self.get_stats()
        cleaned_items = {
            'vector_cache': 0,
            'similarity_cache': 0,
            'memory_freed': 0
        }
        
        try:
            # Calcul de la taille actuelle totale
            total_cache_mb = (
                stats_before['memory_usage']['vector_cache_mb'] +
                stats_before['memory_usage']['similarity_cache_mb']
            )
            
            self.logger.info(f"Taille cache actuelle: {total_cache_mb:.1f}MB")
            
            if total_cache_mb > max_cache_size_mb:
                # 1. Nettoyage du cache de similarité
                similarity_size = len(self.similarity_cache)
                self.similarity_cache.clear()
                cleaned_items['similarity_cache'] = similarity_size
                
                # 2. Nettoyage sélectif du cache vectoriel
                if stats_before['memory_usage']['vector_cache_mb'] > max_cache_size_mb * 0.8:
                    vector_size = len(self.vector_cache)
                    
                    # Garde uniquement les vecteurs des mots-clés les plus utilisés
                    keep_keywords = set(sorted(
                        self.keyword_data.keys(),
                        key=lambda k: self.keyword_data[k].volume,
                        reverse=True
                    )[:1000])  # Garde les 1000 mots-clés les plus volumineux
                    
                    # Nouveau cache vectoriel
                    new_vector_cache = {
                        k: v for k, v in self.vector_cache.items()
                        if k in keep_keywords
                    }
                    
                    # Mise à jour du cache
                    self.vector_cache = new_vector_cache
                    cleaned_items['vector_cache'] = vector_size - len(new_vector_cache)
                
                # 3. Nettoyage du cache du gestionnaire
                self.cache_manager.clear()
                
                # 4. Force le garbage collector
                import gc
                gc.collect()
                
                # Calcul des statistiques après nettoyage
                stats_after = self.get_stats()
                cleaned_items['memory_freed'] = round(
                    total_cache_mb - (
                        stats_after['memory_usage']['vector_cache_mb'] +
                        stats_after['memory_usage']['similarity_cache_mb']
                    ),
                    2
                )
                
                self.logger.info(
                    f"Nettoyage terminé: {cleaned_items['memory_freed']}MB libérés"
                )
                
            else:
                self.logger.info("Nettoyage non nécessaire")
                
        except Exception as e:
            self.logger.error(f"Erreur pendant le nettoyage: {e}")
            raise
            
        return cleaned_items

# Tests unitaires et exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd
    
    def test_semantic_core():
        """Test des fonctionnalités principales"""
        print("🧪 Démarrage des tests...")
        
        # Initialisation
        with Timer("Initialisation du core"):
            core = LightweightSemanticCore()
        
        # Données de test
        test_data = pd.DataFrame({
            'keyword': ['marketing digital', 'seo', 'référencement naturel', 
                       'stratégie marketing', 'content marketing'],
            'volume': [1000, 800, 900, 600, 700]
        })
        
        # Test du chargement
        print("\n📥 Test du chargement des données...")
        core.load_keywords(test_data)
        assert len(core.keyword_data) == len(test_data), "Erreur de chargement"
        print("✅ Chargement OK")
        
        # Test de similarité
        print("\n🔍 Test du calcul de similarité...")
        similarity = core.calculate_similarity('marketing digital', 'seo')
        print(f"Similarité marketing digital/seo: {similarity:.3f}")
        assert 0 <= similarity <= 1, "Score de similarité invalide"
        print("✅ Calcul de similarité OK")
        
        # Test de recherche
        print("\n🔎 Test de recherche de mots-clés similaires...")
        similar = core.find_similar_keywords('marketing digital', max_results=3)
        print("Mots-clés similaires à 'marketing digital':")
        for kw in similar:
            print(f"- {kw['keyword']}: {kw['similarity']:.3f}")
        assert len(similar) <= 3, "Trop de résultats retournés"
        print("✅ Recherche OK")
        
        # Test d'analyse
        print("\n📊 Test d'analyse de groupe...")
        analysis = core.analyze_keyword_group({'marketing digital', 'seo'})
        assert len(analysis) == 2, "Erreur dans l'analyse de groupe"
        print("✅ Analyse OK")
        
        # Test d'export
        print("\n📤 Test d'export...")
        export = core.export_cocoon({'marketing digital', 'seo'})
        assert 'metadata' in export, "Format d'export invalide"
        print("✅ Export OK")
        
        # Test de nettoyage des caches
        print("\n🧹 Test du nettoyage des caches...")
        core.cleanup_caches(max_cache_size_mb=100)
        print("✅ Nettoyage OK")
        
        # Statistiques finales
        print("\n📈 Statistiques du core:")
        stats = core.get_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subvalue in value.items():
                    print(f"  - {subkey}: {subvalue}")
            else:
                print(f"- {key}: {value}")
        
        print("\n✨ Tous les tests sont passés avec succès!")

    try:
        test_semantic_core()
    except Exception as e:
        print(f"\n❌ Erreur pendant les tests: {str(e)}")
        raise