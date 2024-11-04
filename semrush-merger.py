import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
import re
from unidecode import unidecode
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, partial
import jellyfish
from tqdm import tqdm
import multiprocessing as mp
from itertools import combinations
import os

warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_worker():
    """Initialisation des workers pour le multiprocessing"""
    import numpy as np
    np.seterr(all='ignore')

@lru_cache(maxsize=100000)
def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Calcul optimisé des n-grammes avec cache"""
    if not text or len(text) < n:
        return set()
    return set(text[i:i+n].lower() for i in range(len(text)-n+1))

@lru_cache(maxsize=100000)
def calculate_similarity(kw1: str, kw2: str) -> float:
    """Calcul de similarité avec cache"""
    return jellyfish.jaro_winkler_similarity(kw1, kw2)

def process_chunk(chunk_data: Tuple[np.ndarray, float, Dict]) -> Set[str]:
    """Traitement optimisé d'un chunk de mots-clés"""
    keywords, threshold, already_processed = chunk_data
    final_keywords = set()
    
    # Pré-calcul des n-grammes
    ngrams_cache = {
        kw: get_ngrams(kw) 
        for kw in keywords 
        if kw not in already_processed
    }
    
    # Création d'une matrice de similarité pour le batch
    for kw1, ngrams1 in ngrams_cache.items():
        if kw1 in already_processed:
            continue
            
        is_unique = True
        for kw2 in final_keywords:
            ngrams2 = ngrams_cache.get(kw2, set())
            
            # Calcul rapide de similarité Jaccard
            if ngrams1 and ngrams2:
                jaccard = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
                if jaccard >= 0.95:
                    # Vérification plus précise uniquement si nécessaire
                    if calculate_similarity(kw1, kw2) >= threshold:
                        is_unique = False
                        break
            
        if is_unique:
            final_keywords.add(kw1)
    
    return final_keywords

class SemrushMerger:
    def __init__(self, export_dir: str = './exports', threshold: float = 0.95):
        self.export_dir = Path(export_dir)
        self.threshold = threshold
        self.num_processes = min(mp.cpu_count(), 8)  # Limite pour éviter la surcharge
        self.chunk_size = min(10000, os.cpu_count() * 1000)
        self._clean_pattern = re.compile(r'[^a-zA-Z0-9àâäéèêëîïôöùûüçñ\s-]')
        self._processed_keywords = set()

    def clean_text(self, text: str) -> str:
        """Nettoyage optimisé du texte"""
        if not isinstance(text, str):
            return ""
        return ' '.join(self._clean_pattern.sub('', text.lower()).split()).strip()

    def read_excel_optimized(self, file: Path) -> Optional[pd.DataFrame]:
        """Lecture optimisée des fichiers Excel"""
        try:
            df = pd.read_excel(
                file, 
                usecols=['Keyword', 'Search Volume'],
                engine='openpyxl'
            )
            
            if df.empty:
                return None
                
            df.columns = ['keyword', 'volume']
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['keyword'] = df['keyword'].apply(self.clean_text)
            
            mask = (df['keyword'].str.len() > 0) & (df['volume'] > 0)
            df = df[mask]
            
            logger.info(f"Lu {len(df)} lignes depuis {file.name}")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lecture {file.name}: {e}")
            return None

    def process_files_parallel(self, files: List[Path]) -> pd.DataFrame:
        """Traitement parallèle optimisé des fichiers avec déduplication précoce"""
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [executor.submit(self.read_excel_optimized, f) for f in files]
            dfs = []
            
            for future in tqdm(futures, desc="Lecture des fichiers"):
                result = future.result()
                if result is not None and not result.empty:
                    # Déduplication au niveau de chaque fichier
                    result = result.drop_duplicates(subset=['keyword'], keep='first')
                    dfs.append(result)
            
        if not dfs:
            raise ValueError("Aucune donnée valide trouvée")
        
        # Fusion des DataFrames
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Déduplication globale après la fusion
        logger.info(f"Total avant déduplication: {len(merged_df):,}")
        merged_df = merged_df.sort_values('volume', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=['keyword'], keep='first')
        logger.info(f"Total après déduplication: {len(merged_df):,}")
        
        return merged_df


    def find_similar_keywords_parallel(self, keywords: np.ndarray) -> Set[str]:
        """Recherche parallèle optimisée des mots-clés similaires"""
        chunks = np.array_split(keywords, self.num_processes * 4)
        chunk_data = [
            (chunk, self.threshold, self._processed_keywords.copy()) 
            for chunk in chunks
        ]
        
        with ProcessPoolExecutor(
            max_workers=self.num_processes, 
            initializer=init_worker
        ) as executor:
            results = list(tqdm(
                executor.map(process_chunk, chunk_data),
                total=len(chunks),
                desc="Analyse sémantique"
            ))
        
        return set().union(*results)

    def merge_files(self) -> pd.DataFrame:
        """Fusion optimisée des fichiers avec déduplication précoce"""
        excel_files = list(self.export_dir.glob('*.xlsx'))
        if not excel_files:
            raise FileNotFoundError("Aucun fichier Excel trouvé")
        
        logger.info(f"Traitement de {len(excel_files)} fichiers sur {self.num_processes} processeurs")
        
        # Lecture, fusion et déduplication
        merged_df = self.process_files_parallel(excel_files)
        total_initial = len(merged_df)
        logger.info(f"Total après déduplication exacte: {total_initial:,} entrées")
        
        # Optimisation mémoire pour les très gros jeux de données
        if total_initial > 1_000_000:
            logger.warning("Large dataset détecté, échantillonnage à 1M lignes")
            merged_df = merged_df.sample(n=1_000_000, random_state=42)
        
        # Traitement des similarités sur les mots-clés uniques
        unique_keywords = merged_df['keyword'].unique()
        logger.info(f"Analyse sémantique sur {len(unique_keywords):,} mots-clés uniques")
        
        final_keywords = self.find_similar_keywords_parallel(unique_keywords)
        
        # Création du DataFrame final
        final_df = merged_df[merged_df['keyword'].isin(final_keywords)]
        final_df = final_df.sort_values('volume', ascending=False)
        final_df = final_df.reset_index(drop=True)
        
        logger.info(f"Total final après analyse sémantique: {len(final_df):,} entrées")
        return final_df

    def save_results(self, df: pd.DataFrame, output_file: str = 'keywords.csv'):
        """Sauvegarde optimisée des résultats"""
        output_path = self.export_dir / output_file
        
        # Calcul des statistiques
        stats = {
            'total': len(df),
            'total_volume': df['volume'].sum(),
            'mean_volume': df['volume'].mean(),
            'median_volume': df['volume'].median(),
            'top5': df.nlargest(5, 'volume')[['keyword', 'volume']].values.tolist()
        }
        
        # Sauvegarde
        df.to_csv(output_path, index=False)
        logger.info(f"Résultats sauvegardés dans {output_path}")
        
        self._display_stats(stats)

    def _display_stats(self, stats: Dict):
        """Affichage des statistiques"""
        print(f"\nStatistiques de l'analyse :")
        print(f"{'='*40}")
        print(f"Mots-clés uniques   : {stats['total']:,}")
        print(f"Volume total        : {stats['total_volume']:,}")
        print(f"Volume moyen        : {stats['mean_volume']:,.0f}")
        print(f"Volume médian       : {stats['median_volume']:,.0f}")
        print(f"\nTop 5 des mots-clés par volume :")
        print(f"{'-'*40}")
        for kw, vol in stats['top5']:
            print(f"{kw:<30} : {vol:,}")

def main():
    try:
        merger = SemrushMerger()
        final_df = merger.merge_files()
        merger.save_results(final_df)
    except Exception as e:
        logger.error(f"Erreur critique : {e}")
        raise

if __name__ == "__main__":
    main()