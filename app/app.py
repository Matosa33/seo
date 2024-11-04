"""
Application Streamlit optimisée pour la génération de cocons sémantiques.
Interface utilisateur performante avec gestion d'état optimisée.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go 
import networkx as nx
import json 
from datetime import datetime
import io
from typing import List, Dict, Optional, Set
import plotly.express as px
from core import OptimizedSemanticCore
from utils import Timer, format_number, safe_filename

class SemanticUI:
    """Interface utilisateur optimisée avec gestion d'état centralisée"""
    
    def __init__(self):
        """Initialisation optimisée de l'interface"""
        self._configure_page()
        self._init_state()
        self.analyzer = self._initialize_core()  # Changé ici
        self._setup_styles()

    @staticmethod  # Changé en méthode statique
    @st.cache_resource
    def _initialize_core():
        """Initialisation mise en cache du core sémantique"""
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        return OptimizedSemanticCore(str(cache_dir))

    def _configure_page(self):
        """Configuration optimisée de la page"""
        st.set_page_config(
            page_title="Semantic Analysis",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def _init_state(self):
        """Initialisation optimisée de l'état avec structure unifiée"""
        if 'state' not in st.session_state:
            st.session_state.state = {
                'keywords': [],
                'data': None,
                'results': None,
                'page': 0,
                'needs_update': False,
                'graph': nx.Graph(),
                'settings': {
                    'similarity': 0.3,
                    'volume': 0,
                    'max_suggestions': 10,
                    'mode': "Équilibré",  # Changé ici pour correspondre aux options
                    'details': False
                },
                'search': '',
                'last_update': None
            }

    def _setup_styles(self):
        """Application des styles optimisés"""
        st.markdown("""
        <style>
        /* Base */
        .main {
            background-color: #f8fafc;
        }
        
        /* Cards */
        .card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Metrics */
        .metric {
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #0f172a;
        }
        
        .metric-label {
            color: #64748b;
            font-size: 0.875rem;
        }
        
        /* Actions */
        .stButton>button {
            width: 100%;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
        }
        
        /* Tables */
        .dataframe {
            border: none !important;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe th {
            background: #f1f5f9 !important;
            font-weight: 500 !important;
        }
        
        /* Inputs */
        .stTextInput>div>div>input {
            border-radius: 6px;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: #f1f5f9;
        }
        
        /* Charts */
        .plot-container {
            border-radius: 8px;
            overflow: hidden;
            background: white;
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def _analyze_keywords(self, keywords: frozenset) -> Dict:
        """Analyse mise en cache des mots-clés"""
        return self.analyzer.analyze_keyword_group(keywords)

    def process_file(self, file: io.BytesIO) -> Optional[pd.DataFrame]:
        """Traitement optimisé des fichiers"""
        try:
            df = pd.read_csv(file)
            
            if not {'keyword', 'volume'}.issubset(df.columns):
                st.error("❌ Format incorrect (colonnes requises: keyword, volume)")
                return None
                
            # Nettoyage optimisé
            df['keyword'] = df['keyword'].str.lower().str.strip()
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
            return df.drop_duplicates(subset=['keyword'])
            
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            return None

    def render_sidebar(self):
        """Barre latérale optimisée avec contrôles principaux"""
        with st.sidebar:
            st.title("🎯 Configuration")
            
            # Import
            uploaded = st.file_uploader(
                "Import CSV",
                type=['csv'],
                key='file_upload',
                help="Format: keyword,volume"
            )
            
            if uploaded:
                with st.spinner("Traitement..."):
                    df = self.process_file(uploaded)
                    if df is not None:
                        self.analyzer.load_keywords(df)
                        st.session_state.state['data'] = df
                        st.success(f"✅ {len(df)} mots-clés importés")
            
            st.divider()
            
            # Paramètres
            st.subheader("⚙️ Paramètres")
            
        # Mode sélection avec correspondance de valeur initiale
            mode_options = ["Précis", "Équilibré", "Exploratoire"]
            current_mode = st.session_state.state['settings'].get('mode', "Équilibré")
            if current_mode not in mode_options:
                current_mode = "Équilibré"  # Valeur par défaut sécurisée
                
            mode = st.select_slider(
                "Mode",
                options=mode_options,
                value=current_mode
            )
            
            # Mise à jour des paramètres selon le mode
            mode_params = {
                "Précis": {'similarity': 0.6, 'volume_factor': 1.0},
                "Équilibré": {'similarity': 0.4, 'volume_factor': 0.7},
                "Exploratoire": {'similarity': 0.3, 'volume_factor': 0}
            }
            
            new_params = mode_params[mode]
                
            col1, col2 = st.columns(2)
            
            with col1:
                similarity = st.number_input(
                    "Similarité min",
                    min_value=0.1,
                    max_value=0.9,
                    value=new_params['similarity'],
                    step=0.1,
                    format="%.1f"
                )
                
            with col2:
                new_volume = st.number_input(  # Renommé pour clarté
                    "Volume min",
                    min_value=0,
                    value=int(
                        st.session_state.state['settings']['volume'] * 
                        new_params['volume_factor']
                    ),
                    step=100
                )
                # Mise à jour de l'état
                if new_volume != st.session_state.state['settings']['volume']:
                    st.session_state.state['settings']['volume'] = new_volume
                    st.session_state.state['needs_update'] = True

            # Options avancées dans un expander
            with st.expander("🔧 Options avancées"):
                new_max_suggestions = st.slider(  # Renommé pour clarté
                    "Suggestions max",
                    min_value=5,
                    max_value=50,
                    value=st.session_state.state['settings']['max_suggestions']
                )
                # Mise à jour de l'état
                if new_max_suggestions != st.session_state.state['settings']['max_suggestions']:
                    st.session_state.state['settings']['max_suggestions'] = new_max_suggestions
                    st.session_state.state['needs_update'] = True
                
                new_show_details = st.toggle(  # Renommé pour clarté
                    "Afficher les détails",
                    value=st.session_state.state['settings']['details']
                )
                # Mise à jour de l'état
                if new_show_details != st.session_state.state['settings']['details']:
                    st.session_state.state['settings']['details'] = new_show_details
                    st.session_state.state['needs_update'] = True
            
            # Actions
            st.divider()
            st.subheader("🔄 Actions")
            
            if st.button(
                "Analyser",
                use_container_width=True,
                disabled=not st.session_state.state['keywords']
            ):
                self._perform_analysis()
                
            if st.button(
                "Exporter",
                use_container_width=True,
                disabled=not st.session_state.state['results']
            ):
                self._export_results()
    
    def render_main_content(self):
        """Contenu principal optimisé"""
        if st.session_state.state['data'] is None:
            self._render_welcome()
            return
            
        # Layout principal
        col1, col2 = st.columns([2, 3])
        
        with col1:
            self._render_keyword_selector()
        
        with col2:
            self._render_analysis_view()
    
    def _render_welcome(self):
        """Page d'accueil optimisée"""
        st.header("🎯 Analyse Sémantique")
        
        st.markdown("""
        ### Pour commencer :
        
        1. 📤 Importez votre fichier CSV via le panneau de gauche
        2. 🎯 Sélectionnez vos mots-clés cibles
        3. 📊 Analysez les relations sémantiques
        4. 💾 Exportez vos résultats
        
        #### Format du fichier CSV :
        ```
        keyword,volume
        marketing digital,1000
        seo,800
        référencement,500
        ```
        """)
        
        # Exemple téléchargeable
        example = pd.DataFrame({
            'keyword': ['marketing digital', 'seo', 'référencement'],
            'volume': [1000, 800, 500]
        })
        
        st.download_button(
            "📥 Télécharger l'exemple",
            example.to_csv(index=False),
            "exemple.csv",
            "text/csv"
        )

    @staticmethod  # On rend la méthode statique
    @st.cache_data(ttl=60)
    def _get_filtered_data(df: pd.DataFrame, search: str, page: int, per_page: int):
        """Filtrage et pagination optimisés avec mise en cache"""
        df = df.copy()  # Important : on fait une copie pour éviter les modifications du DataFrame original
        
        if search:
            df = df[df['keyword'].str.contains(search, case=False)]
            
        total_pages = (len(df) + per_page - 1) // per_page
        df = df.sort_values('volume', ascending=False)
        
        start = page * per_page
        end = start + per_page
        
        return df.iloc[start:end], total_pages

    def _render_keyword_selector(self):
        """Sélecteur de mots-clés optimisé pour grand volume"""
        st.subheader("🎯 Sélection des mots-clés")
        
        # Barre de recherche avec filtrage côté client
        search = st.text_input(
            "🔍 Rechercher",
            value=st.session_state.state['search'],
            key='keyword_search'
        )
        
        # Préparation optimisée du DataFrame
        if 'display_df' not in st.session_state:
            # On ne fait cette opération qu'une fois
            display_df = st.session_state.state['data'].copy()
            display_df['volume_fmt'] = display_df['volume'].apply(format_number)
            # On garde seulement les colonnes nécessaires
            display_df = display_df[['keyword', 'volume_fmt']]
            display_df.columns = ['Mot-clé', 'Volume']
            st.session_state.display_df = display_df
        
        # Configuration et affichage optimisé du tableau
        st.dataframe(
            st.session_state.display_df,
            use_container_width=True,
            height=600,  # Hauteur ajustable selon vos besoins
            column_config={
                "Mot-clé": st.column_config.TextColumn(
                    "Mot-clé",
                    width="medium",
                    help="Mot-clé SEO",
                ),
                "Volume": st.column_config.TextColumn(
                    "Volume",
                    width="small",
                    help="Volume de recherche mensuel",
                ),
            },
            # Configuration du filtrage et du tri natif
            column_order=["Mot-clé", "Volume"],
            hide_index=True,
        )

        # Sélection optimisée
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected = st.selectbox(
                "Sélectionner un mot-clé",
                options=st.session_state.display_df['Mot-clé'].tolist(),
                key='keyword_select',
                help="Choisissez un mot-clé à ajouter au cocon"
            )
            
        with col2:
            st.write("")
            st.write("")
            if st.button("Ajouter ➕"):
                if selected not in st.session_state.state['keywords']:
                    st.session_state.state['keywords'].append(selected)
                    st.session_state.state['needs_update'] = True
                    st.rerun()
                
    def _render_analysis_view(self):
        """Vue d'analyse optimisée"""
        st.subheader("📊 Analyse")
        
        if not st.session_state.state['keywords']:
            st.info("👈 Sélectionnez des mots-clés pour commencer l'analyse")
            return
            
        # Onglets d'analyse
        tab1, tab2, tab3 = st.tabs([
            "🔍 Relations",
            "📈 Métriques",
            "🎯 Suggestions"
        ])
        
        with tab1:
            self._render_semantic_network()
            
        with tab2:
            self._render_metrics()
            
        with tab3:
            self._render_suggestions()

    def _render_semantic_network(self):
        """Visualisation optimisée du réseau sémantique"""
        if not st.session_state.state['graph'].nodes():
            if st.session_state.state['needs_update']:
                st.warning("⚠️ Cliquez sur Analyser pour voir les relations")
                return
                
            st.info("ℹ️ Aucune relation à afficher")
            return
            
        # Création du layout
        pos = nx.spring_layout(st.session_state.state['graph'])
        
        # Préparation des données en une seule passe
        node_data = {
            'x': [], 'y': [], 'text': [], 'size': [], 'color': []
        }
        edge_data = {
            'x': [], 'y': [], 'text': []
        }
        
        # Traitement optimisé des nœuds
        max_volume = max(
            d['volume'] for _, d in 
            st.session_state.state['graph'].nodes(data=True)
        )
        
        for node, data in st.session_state.state['graph'].nodes(data=True):
            x, y = pos[node]
            node_data['x'].append(x)
            node_data['y'].append(y)
            node_data['text'].append(f"{node}<br>Volume: {format_number(data['volume'])}")
            node_data['size'].append(20 + (data['volume'] / max_volume) * 50)
            node_data['color'].append(data['volume'])
            
        # Traitement optimisé des liens
        for edge in st.session_state.state['graph'].edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_data['x'].extend([x0, x1, None])
            edge_data['y'].extend([y0, y1, None])
            edge_data['text'].append(f"Similarité: {edge[2]['weight']:.2f}")
        
        # Création optimisée du graphe
        fig = go.Figure()
        
        # Ajout des liens
        fig.add_trace(go.Scatter(
            x=edge_data['x'],
            y=edge_data['y'],
            line=dict(width=0.5, color='#94a3b8'),
            hoverinfo='text',
            text=edge_data['text'],
            mode='lines'
        ))
        
        # Ajout des nœuds
        fig.add_trace(go.Scatter(
            x=node_data['x'],
            y=node_data['y'],
            mode='markers+text',
            hoverinfo='text',
            text=node_data['text'],
            textposition="top center",
            marker=dict(
                size=node_data['size'],
                color=node_data['color'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Volume',
                    thickness=15,
                    len=0.5
                )
            )
        ))
        
        # Configuration optimisée
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_metrics(self):
        """Affichage optimisé des métriques"""
        if not st.session_state.state['results']:
            st.info("ℹ️ Lancez l'analyse pour voir les métriques")
            return
            
        # Calcul optimisé des métriques principales
        metrics = self._calculate_metrics(
            st.session_state.state['results'],
            self.analyzer  # On passe l'analyseur comme paramètre
        )
        
        # Affichage en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._metric_card(
                "Volume Total",
                format_number(metrics['total_volume']),
                "📊"
            )
            
        with col2:
            self._metric_card(
                "Score Moyen",
                f"{metrics['avg_score']:.2f}",
                "⭐"
            )
            
        with col3:
            self._metric_card(
                "Connexions",
                str(metrics['connections']),
                "🔗"
            )
        
        # Graphiques optimisés
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_volume_distribution(metrics['volumes'])
            
        with col2:
            self._render_similarity_matrix(metrics['similarities'])

    def _metric_card(self, title: str, value: str, icon: str):
        """Carte de métrique optimisée"""
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">{icon} {title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    @st.cache_data
    def _calculate_metrics(results: Dict, _analyzer) -> Dict:
        """Calcul optimisé et mis en cache des métriques
        
        Args:
            results: Dictionnaire des résultats
            _analyzer: Instance de OptimizedSemanticCore (préfixé avec _ pour éviter le hachage)
        """
        volumes = [data['volume'] for data in results.values()]
        
        # Matrice de similarité
        keywords = list(results.keys())
        n = len(keywords)
        similarities = np.zeros((n, n))
        
        for i, kw1 in enumerate(keywords):
            for j, kw2 in enumerate(keywords[i+1:], i+1):
                sim = _analyzer.calculate_similarity(kw1, kw2)
                similarities[i, j] = similarities[j, i] = sim
        
        return {
            'total_volume': sum(volumes),
            'avg_score': np.mean([d['semantic_strength'] for d in results.values()]),
            'connections': sum(len(d['related_keywords']) for d in results.values()),
            'volumes': volumes,
            'similarities': similarities
        }


    def _render_metrics(self):
        """Affichage optimisé des métriques"""
        if not st.session_state.state['results']:
            st.info("ℹ️ Lancez l'analyse pour voir les métriques")
            return
            
        # Calcul optimisé des métriques principales
        metrics = self._calculate_metrics(
            results=st.session_state.state['results'],
            _analyzer=self.analyzer  # Passage nommé avec underscore
        )

    def _render_similarity_matrix(self, similarities: np.ndarray):
        """Matrice de similarité optimisée"""
        fig = px.imshow(
            similarities,
            title="Matrice de similarité",
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_suggestions(self):
        """Interface optimisée pour les suggestions sémantiques"""
        if not st.session_state.state['results']:
            st.info("ℹ️ Lancez l'analyse pour voir les suggestions")
            return
            
        # En-tête avec métriques globales
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 🎯 Suggestions Sémantiques")
            with col2:
                mode = st.selectbox(
                    "Mode d'analyse",
                    ["Équilibré", "Sémantique", "Volume", "Diversité"],
                    key='suggestion_mode'
                )

        # Contrôles avancés
        with st.expander("⚙️ Options avancées", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                min_similarity = st.slider(
                    "Similarité minimum",
                    0.0, 1.0, 0.3, 0.05,
                    key='min_similarity'
                )
            with col2:
                min_volume = st.number_input(
                    "Volume minimum",
                    0, 100000, 0,
                    key='min_volume'
                )
            with col3:
                max_suggestions = st.slider(
                    "Nombre de suggestions",
                    5, 50, 10,
                    key='max_suggestions'
                )

        # Filtrage intelligent
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input(
                "🔍 Filtrer les suggestions",
                key='suggestion_search'
            )
        with col2:
            sort_by = st.selectbox(
                "Trier par",
                ["Pertinence", "Score Sémantique", "Volume", "Diversité"],
                key='sort_suggestions'
            )

        # Récupération des suggestions avec nouveaux paramètres
        suggestions = self._get_filtered_suggestions(
            suggestions=st.session_state.state.get('suggestions', []),
            search=search,
            sort_by=sort_by,
            mode=mode
        )

        # Affichage amélioré des suggestions
        for sugg in suggestions:
            with st.container():
                cols = st.columns([3, 1, 1, 1, 1])
                
                with cols[0]:
                    st.markdown(f"**{sugg['keyword']}**")
                    if st.session_state.state['settings']['details']:
                        st.caption(f"Connexions: {sugg['direct_connections']} directes, {sugg['secondary_connections']} secondaires")
                
                with cols[1]:
                    score_color = self._get_score_color(sugg['relevance'])
                    st.markdown(f"<span style='color:{score_color}'>Score: {sugg['relevance']:.2f}</span>", unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown(f"Similarité: {sugg['similarity']:.2f}")
                
                with cols[3]:
                    st.markdown(f"Volume: {format_number(sugg['volume'])}")
                
                with cols[4]:
                    if st.button("Ajouter ➕", key=f"add_sugg_{sugg['keyword']}"):
                        if sugg['keyword'] not in st.session_state.state['keywords']:
                            st.session_state.state['keywords'].append(sugg['keyword'])
                            st.session_state.state['needs_update'] = True
                            st.rerun()

    @staticmethod
    @st.cache_data
    def _get_filtered_suggestions(suggestions: List[Dict], search: str, sort_by: str, mode: str) -> List[Dict]:
        """Filtrage optimisé des suggestions avec mise en cache"""
        # Copie pour éviter la modification des données d'origine
        filtered_suggestions = suggestions.copy()
        
        # Filtrage par recherche
        if search:
            filtered_suggestions = [
                s for s in filtered_suggestions 
                if search.lower() in s['keyword'].lower()
            ]
        
        # Application du mode d'analyse
        for sugg in filtered_suggestions:
            if mode == "Sémantique":
                sugg['sort_score'] = sugg['similarity'] * 0.7 + sugg.get('centroid_similarity', 0) * 0.3
            elif mode == "Volume":
                sugg['sort_score'] = sugg['volume']
            elif mode == "Diversité":
                sugg['sort_score'] = (
                    sugg['relevance'] * 0.6 + 
                    (1 - sugg.get('direct_connections', 0) / 10) * 0.4
                )
            else:  # Mode Équilibré
                sugg['sort_score'] = sugg['relevance']

        # Tri optimisé
        sort_keys = {
            "Pertinence": lambda x: (-x['relevance'], -x['volume']),
            "Score Sémantique": lambda x: (-x['similarity'], -x['volume']),
            "Volume": lambda x: (-x['volume'], -x['sort_score']),
            "Diversité": lambda x: (-x['sort_score'], -x['volume'])
        }
        
        return sorted(
            filtered_suggestions,
            key=sort_keys.get(sort_by, sort_keys["Pertinence"])
        )
        
    @staticmethod
    def _get_score_color(score: float) -> str:
        """Retourne une couleur basée sur le score"""
        if score >= 0.8:
            return "#2E7D32"  # Vert foncé
        elif score >= 0.6:
            return "#4CAF50"  # Vert
        elif score >= 0.4:
            return "#FFA726"  # Orange
        elif score >= 0.2:
            return "#EF6C00"  # Orange foncé
        else:
            return "#D32F2F"  # Rouge


    def _perform_analysis(self):
        """Exécution optimisée de l'analyse"""
        with st.spinner("Analyse en cours..."):
            try:
                # Analyse du groupe
                results = self.analyzer.analyze_keyword_group(
                    set(st.session_state.state['keywords']),
                    min_similarity=st.session_state.state['settings']['similarity']
                )
                
                # Génération des suggestions
                suggestions = self.analyzer.suggest_keywords(
                    current_keywords=set(st.session_state.state['keywords']),
                    max_suggestions=st.session_state.state['settings']['max_suggestions'],
                    min_similarity=st.session_state.state['settings']['similarity'],
                    min_volume=st.session_state.state['settings']['volume']
                )
                
                # Mise à jour du graphe
                self._update_graph(results)
                
                # Sauvegarde des résultats et suggestions
                st.session_state.state['results'] = results
                st.session_state.state['suggestions'] = suggestions
                st.session_state.state['needs_update'] = False
                st.session_state.state['last_update'] = datetime.now().isoformat()
                
                st.success("✅ Analyse terminée")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")

    def _update_graph(self, results: Dict):
        """Mise à jour optimisée du graphe"""
        G = nx.Graph()
        
        # Ajout des nœuds
        for kw, data in results.items():
            G.add_node(kw, volume=data['volume'])
        
        # Ajout des liens
        for kw, data in results.items():
            for rel in data['related_keywords']:
                if rel['similarity'] >= st.session_state.state['settings']['similarity']:
                    G.add_edge(kw, rel['keyword'], weight=rel['similarity'])
        
        st.session_state.state['graph'] = G

    def _export_results(self):
        """Export optimisé des résultats"""
        if not st.session_state.state['results']:
            st.error("❌ Aucun résultat à exporter")
            return
            
        try:
            # Préparation des données
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'keywords': st.session_state.state['keywords'],
                    'settings': st.session_state.state['settings']
                },
                'results': st.session_state.state['results']
            }
            
            # Export JSON
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            
            st.download_button(
                "📥 Télécharger les résultats",
                json_str,
                f"analyse_{datetime.now():%Y%m%d_%H%M}.json",
                "application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export: {str(e)}")

    def run(self):
        """Point d'entrée principal optimisé"""
        try:
            self.render_sidebar()
            self.render_main_content()
            
        except Exception as e:
            st.error("❌ Erreur critique")
            st.exception(e)


def main():
    """Fonction principale avec gestion d'erreurs"""
    try:
        ui = SemanticUI()
        ui.run()
    except Exception as e:
        st.error("❌ Erreur fatale de l'application")
        st.exception(e)

if __name__ == "__main__":
    main()