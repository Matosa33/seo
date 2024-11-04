"""
Application Streamlit pour la génération de cocons sémantiques.
Interface utilisateur moderne et optimisée.
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
from typing import List, Dict, Tuple, Optional
import plotly.express as px
from core import LightweightSemanticCore
from utils import Timer, format_number, safe_filename

class ModernSemanticUI:
    """Interface utilisateur moderne pour le générateur de cocons"""
    
    def __init__(self):
        """Initialisation de l'interface"""
        self._configure_page()
        self._setup_theme()
        self._init_core()
        self._init_session()

    def _configure_page(self):
        """Configuration de la page Streamlit"""
        st.set_page_config(
            page_title="Semantic Cocoon Generator",
            page_icon="🕸️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def _setup_theme(self):
        """Configuration du thème et des styles"""
        st.markdown("""
        <style>
        /* Style global */
        body {
            font-family: 'Inter', sans-serif;
            color: #1E293B;
        }
        
        /* Cards */
        div.stCard {
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            padding: 1.5rem;
            background: white;
            margin-bottom: 1rem;
        }
        
        /* Badges */
        .badge {
            background: #E2E8F0;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            color: #475569;
            display: inline-block;
            margin: 0.25rem;
        }
        
        .badge-primary {
            background: #EDE9FE;
            color: #6D28D9;
        }
        
        .badge-success {
            background: #DCFCE7;
            color: #15803D;
        }
        
        /* Stats */
        .stat-card {
            text-align: center;
            padding: 1rem;
            background: #F8FAFC;
            border-radius: 8px;
            margin: 0.5rem;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1E293B;
        }
        
        .stat-label {
            font-size: 0.875rem;
            color: #64748B;
        }
        
        /* Tables */
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .styled-table th {
            background: #F1F5F9;
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
        }
        
        .styled-table td {
            padding: 0.75rem 1rem;
            border-top: 1px solid #E2E8F0;
        }
        
        /* Graphiques */
        .plotly-chart {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Boutons */
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Inputs */
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            padding: 0.75rem 1rem;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background: #F8FAFC;
        }
        </style>
        """, unsafe_allow_html=True)

    def _init_core(self):
        """Initialisation du core sémantique"""
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        self.analyzer = LightweightSemanticCore(
            cache_dir=str(cache_dir)
        )

    def _init_session(self):
        """Initialisation des variables de session"""
        if 'selected_keywords' not in st.session_state:
            st.session_state.selected_keywords = []
        if 'min_similarity' not in st.session_state:
            st.session_state.min_similarity = 0.3
        if 'min_volume' not in st.session_state:
            st.session_state.min_volume = 0
        if 'graph' not in st.session_state:
            st.session_state.graph = nx.Graph()

    def render_sidebar(self):
            """Affichage de la barre latérale améliorée"""
            with st.sidebar:
                st.markdown("## 🎯 Configuration de l'analyse")
                
                # Section Import
                st.markdown("#### 📤 Import de données")
                uploaded_file = st.file_uploader(
                    "Fichier CSV de mots-clés",
                    type=['csv'],
                    help="Format: keyword,volume"
                )
                
                if uploaded_file:
                    self.process_uploaded_file(uploaded_file)
                
                if st.checkbox("Voir un exemple"):
                    st.code("""keyword,volume
    marketing digital,1000
    seo,800
    référencement naturel,900""")
                    
                    example_df = pd.DataFrame({
                        'keyword': ['marketing digital', 'seo', 'référencement naturel'],
                        'volume': [1000, 800, 900]
                    })
                    st.download_button(
                        "📥 Télécharger l'exemple",
                        example_df.to_csv(index=False),
                        "exemple_mots_cles.csv",
                        "text/csv"
                    )
                
                st.divider()
                
                # Section Configuration
                st.markdown("#### ⚙️ Paramètres")
                
                # Mode de suggestion avec indicateurs visuels
                st.markdown("##### Mode de suggestion")
                mode = st.select_slider(
                    "Mode de suggestion",
                    options=["Précis", "Équilibré", "Exploratoire"],
                    value="Équilibré",
                    label_visibility="collapsed"  # Cache l'étiquette visuellement mais la garde pour l'accessibilité
                )
                
                # Ajustement automatique des paramètres selon le mode
                if mode == "Précis":
                    default_similarity = 0.6
                    default_volume = st.session_state.min_volume
                elif mode == "Équilibré":
                    default_similarity = 0.4
                    default_volume = int(st.session_state.min_volume * 0.7)
                else:
                    default_similarity = 0.3
                    default_volume = 0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    similarity = st.number_input(
                        "Similarité min",
                        min_value=0.1,
                        max_value=0.9,
                        value=default_similarity,
                        step=0.1,
                        format="%.1f"
                    )
                
                with col2:
                    volume = st.number_input(
                        "Volume min",
                        min_value=0,
                        value=default_volume,
                        step=100
                    )
                
                st.session_state.min_similarity = similarity
                st.session_state.min_volume = volume
                
                # Options avancées
                st.markdown("##### 🔧 Options avancées")
                max_suggestions = st.slider(
                    "Nombre de suggestions",
                    min_value=5,
                    max_value=50,
                    value=10
                )
                st.session_state.max_suggestions = max_suggestions
                
                cleanup_threshold = st.slider(
                    "Seuil de nettoyage cache (MB)",
                    min_value=100,
                    max_value=2000,
                    value=1000
                )
                
                if st.button("🧹 Nettoyer le cache"):
                    try:
                        with st.spinner("Nettoyage en cours..."):
                            cleanup_stats = self.analyzer.cleanup_caches(cleanup_threshold)
                            
                            # Affichage des résultats du nettoyage
                            st.success(
                                f"""✅ Cache nettoyé avec succès :
                                - {cleanup_stats['memory_freed']:.1f}MB libérés
                                - {cleanup_stats['vector_cache']} vecteurs supprimés
                                - {cleanup_stats['similarity_cache']} similarités supprimées"""
                            )
                    except Exception as e:
                        st.error(f"❌ Erreur lors du nettoyage : {str(e)}")
                
                # Statistiques
                if hasattr(self.analyzer, 'keyword_data'):
                    st.markdown("#### 📊 Statistiques")
                    stats = self.analyzer.get_stats()
                    
                    # Métriques principales
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Mots-clés",
                            format_number(stats['total_keywords'])
                        )
                    with col2:
                        st.metric(
                            "Volume total",
                            format_number(stats['total_volume'])
                        )
                    
                    # Usage mémoire
                    st.markdown("##### 💾 Utilisation mémoire")
                    memory_usage = stats['memory_usage']
                    total_mb = sum(memory_usage.values())
                    
                    # Graphique d'utilisation mémoire
                    fig = px.pie(
                        values=list(memory_usage.values()),
                        names=list(memory_usage.keys()),
                        title="Répartition mémoire"
                    )
                    fig.update_layout(height=200, margin=dict(t=20, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"Total: **{total_mb:.1f} MB**")
                
                st.divider()
                
                # Actions
                st.markdown("#### 🛠️ Actions")
                if st.button("💾 Exporter le cocon", use_container_width=True):
                    if not st.session_state.selected_keywords:
                        st.error("❌ Aucun mot-clé sélectionné")
                    else:
                        with st.spinner("Export en cours..."):
                            data = self.analyzer.export_cocoon(
                                set(st.session_state.selected_keywords)
                            )
                            
                            # JSON formaté
                            json_str = json.dumps(
                                data,
                                ensure_ascii=False,
                                indent=2
                            )
                            
                            st.download_button(
                                "📥 Télécharger le JSON",
                                json_str,
                                f"cocon_{datetime.now():%Y%m%d_%H%M}.json",
                                "application/json"
                            )

    def render_main_content(self):
        """Affichage du contenu principal"""
        if 'keywords_df' not in st.session_state:
            self.render_welcome_screen()
            return
            
        # Layout principal
        col1, col2 = st.columns([2, 3])
        
        with col1:
            self.render_keyword_selector()
        
        with col2:
            self.render_cocoon_visualizer()

    def render_welcome_screen(self):
        """Affichage de l'écran de bienvenue"""
        st.markdown("""
        # 🕸️ Générateur de Cocons Sémantiques
        
        Créez des cocons sémantiques optimisés pour le SEO en analysant les relations 
        entre vos mots-clés.
        
        ### 🚀 Pour commencer
        
        1. Préparez votre fichier CSV avec les colonnes :
           - `keyword` : vos mots-clés
           - `volume` : volume de recherche mensuel
        
        2. Importez le fichier via le panneau de gauche
        
        3. Sélectionnez vos premiers mots-clés pour construire votre cocon
        
        ### 📊 Fonctionnalités
        
        - Analyse sémantique avancée
        - Suggestions intelligentes
        - Visualisation interactive
        - Export des données
        """)
        
        # Exemple minimaliste
        with st.expander("👉 Voir un exemple de fichier"):
            st.code("""keyword,volume
marketing digital,1000
seo,800
référencement naturel,900""")
            
            example_df = pd.DataFrame({
                'keyword': ['marketing digital', 'seo', 'référencement naturel'],
                'volume': [1000, 800, 900]
            })
            
            st.download_button(
                "📥 Télécharger l'exemple",
                example_df.to_csv(index=False),
                "exemple_mots_cles.csv",
                "text/csv"
            )

    def process_uploaded_file(self, file: io.BytesIO):
        """Traitement du fichier uploadé"""
        try:
            # Lecture avec gestion des encodages
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='latin1')
            
            # Validation des colonnes
            required_cols = {'keyword', 'volume'}
            if not all(col in df.columns for col in required_cols):
                st.error("❌ Format incorrect. Colonnes requises: keyword, volume")
                return
            
            # Nettoyage des données
            df['keyword'] = df['keyword'].str.lower().str.strip()
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
            df = df.drop_duplicates(subset=['keyword'])
            
            # Chargement dans le core
            with st.spinner("Chargement des données..."):
                self.analyzer.load_keywords(df)
                st.session_state.keywords_df = df
                
            st.success(f"✅ {len(df):,} mots-clés chargés")
            
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

    def render_keyword_selector(self):
        """Sélecteur de mots-clés amélioré"""
        st.markdown("### 🎯 Sélection des mots-clés")
        
        # Recherche et filtrage
        search = st.text_input(
            "🔍 Rechercher",
            help="Filtrer les mots-clés"
        )
        
        # Filtrage des données
        df = st.session_state.keywords_df
        if search:
            df = df[df['keyword'].str.contains(search, case=False)]
        
        # Tri par volume
        df = df# Tri par volume
        df = df.sort_values('volume', ascending=False)
        
        # Affichage du tableau avec mise en forme
        st.dataframe(
            df.style.background_gradient(
                subset=['volume'],
                cmap='YlOrRd'
            ).format({
                'volume': lambda x: f"{x:,}"
            }),
            height=300,
            use_container_width=True
        )
        
        # Interface de sélection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox(
                "Sélectionner un mot-clé",
                options=df['keyword'].tolist(),
                help="Choisir un mot-clé à ajouter au cocon"
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("➕ Ajouter", help="Ajouter au cocon"):
                self.add_keyword_to_cocoon(selected)
        
        # Liste des mots-clés sélectionnés
        if st.session_state.selected_keywords:
            st.markdown("### 📝 Mots-clés sélectionnés")
            
            # Analyse du groupe
            with st.spinner("Analyse en cours..."):
                analysis = self.analyzer.analyze_keyword_group(
                    set(st.session_state.selected_keywords),
                    min_similarity=st.session_state.min_similarity
                )
            
            # Affichage des mots-clés sélectionnés avec métriques
            for kw in st.session_state.selected_keywords:
                with st.container():
                    metrics = analysis.get(kw, {})
                    
                    # Carte de mot-clé
                    st.markdown(f"""
                    <div class="stCard">
                        <h4>{kw}</h4>
                        <div class="badge badge-primary">Volume: {format_number(metrics.get('volume', 0))}</div>
                        <div class="badge badge-success">Force: {metrics.get('semantic_strength', 0):.2f}</div>
                        <div class="badge">Connexions: {len(metrics.get('related_keywords', []))}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Boutons d'action
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("❌", key=f"remove_{kw}"):
                            st.session_state.selected_keywords.remove(kw)
                            st.rerun()
                    
                    with col2:
                        if st.button("📊 Détails", key=f"details_{kw}"):
                            with st.expander("Relations sémantiques"):
                                for rel in metrics.get('related_keywords', [])[:5]:
                                    st.markdown(f"""
                                    <div class="badge">
                                        {rel['keyword']} ({rel['similarity']:.2f})
                                    </div>
                                    """, unsafe_allow_html=True)

    def render_cocoon_visualizer(self):
        """Visualisation du cocon et suggestions"""
        st.markdown("### 🌐 Analyse du cocon")
        
        tabs = st.tabs(["📊 Graphe", "✨ Suggestions", "📈 Statistiques"])
        
        with tabs[0]:
            self.render_semantic_network()
            
        with tabs[1]:
            self.render_suggestions()
            
        with tabs[2]:
            self.render_cocoon_stats()

    def render_semantic_network(self):
        """Visualisation du réseau sémantique"""
        if not st.session_state.selected_keywords:
            st.info("👆 Ajoutez des mots-clés pour voir le réseau")
            return
            
        # Mise à jour du graphe
        self.update_semantic_network()
        
        if len(st.session_state.graph.nodes) == 0:
            return
            
        # Création du layout
        pos = nx.spring_layout(st.session_state.graph, k=1.5)
        
        # Préparation des données
        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        edge_x, edge_y, edge_text = [], [], []
        
        # Traitement des nœuds
        volumes = list(nx.get_node_attributes(st.session_state.graph, 'volume').values())
        max_volume = max(volumes) if volumes else 1
        
        for node, coords in pos.items():
            node_x.append(coords[0])
            node_y.append(coords[1])
            
            volume = st.session_state.graph.nodes[node]['volume']
            node_text.append(f"{node}<br>Volume: {format_number(volume)}")
            node_size.append(20 + (volume / max_volume) * 50)
            node_color.append(volume)
        
        # Traitement des liens
        for edge in st.session_state.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = st.session_state.graph.edges[edge]['weight']
            edge_text.append(f"Similarité: {weight:.2f}")
        
        # Création du graphe
        fig = go.Figure()
        
        # Ajout des liens
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#90A4AE'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        ))
        
        # Ajout des nœuds
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                line=dict(width=1, color='#FFFFFF'),
                colorbar=dict(
                    title='Volume',
                    thickness=15,
                    len=0.5
                )
            )
        ))
        
        # Configuration du layout
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_suggestions(self):
            """Affichage des suggestions amélioré avec analyses avancées"""
            if not st.session_state.selected_keywords:
                st.info("👆 Ajoutez des mots-clés pour voir les suggestions")
                return
            
            # Configuration avancée
            with st.expander("⚙️ Configuration avancée des suggestions"):
                tabs = st.tabs(["Filtres", "Optimisation", "Analyse"])
                
                with tabs[0]:
                    col1, col2 = st.columns(2)
                    with col1:
                        min_distance = st.slider(
                            "Distance minimale",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.05,
                            help="Distance minimale entre les suggestions"
                        )
                    with col2:
                        cluster_threshold = st.slider(
                            "Seuil de clustering",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.5,
                            help="Seuil de regroupement thématique"
                        )
                    
                    diversify = st.checkbox(
                        "Favoriser la diversité",
                        value=True,
                        help="Privilégie des suggestions plus variées"
                    )
                    
                with tabs[1]:
                    col1, col2 = st.columns(2)
                    with col1:
                        intent_weight = st.slider(
                            "Poids de l'intention",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.3,
                            help="Importance de la correspondance d'intention"
                        )
                    with col2:
                        volume_weight = st.slider(
                            "Poids du volume",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.3,
                            help="Importance du volume de recherche"
                        )
                    
                with tabs[2]:
                    show_clusters = st.checkbox(
                        "Afficher les groupes thématiques",
                        value=True
                    )
                    show_predictions = st.checkbox(
                        "Afficher les prédictions de difficulté",
                        value=True
                    )
            
            # Options de filtrage principales
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                search = st.text_input(
                    "🔍 Filtrer les suggestions",
                    help="Rechercher dans les suggestions"
                )
                
            with col2:
                sort_options = ["Pertinence", "Volume", "Similarité"]
                if diversify:
                    sort_options.append("Diversité")
                sort_by = st.selectbox(
                    "Trier par",
                    sort_options
                )
                
            with col3:
                show_details = st.toggle("Détails")
            
            # Récupération des suggestions avec paramètres optimisés
            with st.spinner("Analyse sémantique en cours..."):
                suggestions = self.analyzer.suggest_keywords(
                    current_keywords=set(st.session_state.selected_keywords),
                    max_suggestions=st.session_state.get('max_suggestions', 10),
                    min_similarity=st.session_state.min_similarity,
                    min_volume=st.session_state.min_volume,
                    min_distance=min_distance if diversify else 0.0
                )
                
                if show_clusters and suggestions:
                    clusters = self.analyzer.cluster_suggestions(
                        suggestions,
                        threshold=cluster_threshold
                    )
            
            if not suggestions:
                st.warning("""
                    ⚠️ Aucune suggestion trouvée avec les paramètres actuels.
                    Suggestions d'ajustement :
                    - Réduire le seuil de similarité (actuellement {st.session_state.min_similarity})
                    - Réduire le volume minimum (actuellement {st.session_state.min_volume})
                    - Ajouter plus de mots-clés au cocon
                    - Désactiver le mode diversité si actif
                """)
                return
            
            # Filtrage et tri avancé
            if search:
                suggestions = [s for s in suggestions if search.lower() in s['keyword'].lower()]
                
            # Application des poids personnalisés
            for s in suggestions:
                s['weighted_score'] = (
                    s['similarity'] * (1 - intent_weight - volume_weight) +
                    s['intent_match'] * intent_weight +
                    (s['volume'] / max(s['volume'] for s in suggestions)) * volume_weight
                )
            
            # Tri selon le critère sélectionné
            if sort_by == "Volume":
                suggestions.sort(key=lambda x: x['volume'], reverse=True)
            elif sort_by == "Pertinence":
                suggestions.sort(key=lambda x: x['weighted_score'], reverse=True)
            elif sort_by == "Diversité":
                suggestions.sort(key=lambda x: (x['weighted_score'], x.get('cluster_diversity', 0)), reverse=True)
            else:  # Similarité
                suggestions.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Affichage par clusters si activé
            if show_clusters and 'clusters' in locals():
                for cluster_id, cluster_keywords in clusters.items():
                    with st.expander(f"Groupe thématique {cluster_id + 1}", expanded=True):
                        cluster_suggestions = [s for s in suggestions if s['keyword'] in cluster_keywords]
                        self._render_suggestion_group(cluster_suggestions, show_details, show_predictions)
            else:
                # Affichage normal
                self._render_suggestion_group(suggestions, show_details, show_predictions)

    def _render_suggestion_group(self, suggestions, show_details, show_predictions):
        """Sous-méthode pour l'affichage d'un groupe de suggestions"""
        for suggestion in suggestions:
            with st.container():
                st.markdown(f"""
                <div class="stCard">
                    <h4>{suggestion['keyword']}</h4>
                    <div class="badge badge-primary">Score: {suggestion.get('weighted_score', suggestion['similarity']):.2f}</div>
                    <div class="badge">Volume: {format_number(suggestion['volume'])}</div>
                    <div class="badge badge-success">Pertinence: {suggestion['relevance']:.2f}</div>
                    {"<div class='badge badge-warning'>Difficulté estimée: " + f"{suggestion.get('predicted_difficulty', 'N/A')}</div>" if show_predictions else ""}
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("➕", key=f"add_{suggestion['keyword']}"):
                        self.add_keyword_to_cocoon(suggestion['keyword'])
                
                if show_details:
                    with st.expander("📊 Analyse détaillée"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("🔗 Correspondances")
                            st.markdown(f"""
                            <div class="badge">Matches: {suggestion['matches']}</div>
                            <div class="badge">Intent: {suggestion['intent_match']:.2f}</div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if show_predictions:
                                st.markdown("📈 Prédictions")
                                st.markdown(f"""
                                <div class="badge">Potentiel: {suggestion.get('growth_potential', 'N/A')}</div>
                                <div class="badge">Tendance: {suggestion.get('trend', 'Stable')}</div>
                                """, unsafe_allow_html=True)

    def render_cocoon_stats(self):
        """Affichage des statistiques du cocon"""
        if not st.session_state.selected_keywords:
            st.info("👆 Ajoutez des mots-clés pour voir les statistiques")
            return
            
        # Analyse du groupe
        analysis = self.analyzer.analyze_keyword_group(
            set(st.session_state.selected_keywords)
        )
        
        # Métriques globales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_volume = sum(data['volume'] for data in analysis.values())
            st.metric(
                "Volume total",
                format_number(total_volume)
            )
            
        with col2:
            avg_strength = np.mean([
                data['semantic_strength'] 
                for data in analysis.values()
            ])
            st.metric(
                "Force moyenne",
                f"{avg_strength:.2f}"
            )
            
        with col3:
            connections = sum(
                len(data['related_keywords']) 
                for data in analysis.values()
            )
            st.metric(
                "Connexions",
                connections
            )
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des volumes
            volumes = [data['volume'] for data in analysis.values()]
            fig = px.histogram(
                x=volumes,
                title="Distribution des volumes",
                labels={'x': 'Volume', 'y': 'Nombre'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Force sémantique par mot-clé
            strengths = {
                kw: data['semantic_strength'] 
                for kw, data in analysis.items()
            }
            fig = px.bar(
                x=list(strengths.keys()),
                y=list(strengths.values()),
                title="Force sémantique par mot-clé",
                labels={'x': 'Mot-clé', 'y': 'Force'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    def add_keyword_to_cocoon(self, keyword: str):
        """Ajout d'un mot-clé au cocon avec validation"""
        if keyword in st.session_state.selected_keywords:
            st.warning(f"⚠️ '{keyword}' est déjà dans le cocon")
            return
            
        st.session_state.selected_keywords.append(keyword)
        self.update_semantic_network()
        st.success(f"✅ '{keyword}' ajouté au cocon")

    def update_semantic_network(self):
        """Mise à jour du réseau sémantique"""
        keywords = st.session_state.selected_keywords
        if not keywords:
            return
        
        # Création du nouveau graphe
        G = nx.Graph()
        
        # Ajout des nœuds avec leurs volumes
        for kw in keywords:
            volume = st.session_state.keywords_df[
                st.session_state.keywords_df['keyword'] == kw
            ]['volume'].iloc[0]
            G.add_node(kw, volume=volume)
        
        # Calcul et ajout des liens
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i+1:]:
                similarity = self.analyzer.calculate_similarity(kw1, kw2)
                if similarity > st.session_state.min_similarity:
                    G.add_edge(kw1, kw2, weight=similarity)
        
        st.session_state.graph = G

    def run(self):
        """Point d'entrée principal de l'application"""
        try:
            self.render_sidebar()
            self.render_main_content()
            
        except Exception as e:
            st.error(f"❌ Une erreur est survenue: {str(e)}")
            st.exception(e)

def main():
    """Fonction principale"""
    try:
        app = ModernSemanticUI()
        app.run()
    except Exception as e:
        st.error("❌ Erreur critique de l'application")
        st.exception(e)

if __name__ == "__main__":
    main()