# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 21:08:27 2026

@author: josev
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# Set page config
st.set_page_config(
    page_title="Player Style Clustering Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .cluster-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #90caf9;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("⚽ Player Style Clustering Analysis")
st.markdown("### Unsupervised Machine Learning for Football Player Similarity")

# Sidebar controls
st.sidebar.header("Analysis Controls")
MIN_SHOTS_THRESHOLD = st.sidebar.slider("Minimum Shots Threshold", min_value=20, max_value=100, value=40)
CHOSEN_K = st.sidebar.selectbox("Number of Clusters (K)", options=[4, 5, 6, 7, 8], index=1)

# Show methodology in sidebar
st.sidebar.markdown("---")
st.sidebar.header("📚 Methodology")
st.sidebar.markdown("""
**Cosine K-Means Clustering** is used to group players by style rather than volume.

**Key Features:**
- Shot share (% of team shots)
- Average position (X, Y)
- Situation percentages
- Shot type percentages
- xG metrics
""")

# Main content
st.header("📊 Data Overview")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('datacompleta.parquet')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Unique Players", df['player'].nunique())
    with col3:
        st.metric("Years Covered", f"{df['year'].min()}-{df['year'].max()}")
    with col4:
        st.metric("Total Matches", df['match_id'].nunique())
    
    # Show sample data
    with st.expander("View Sample Data"):
        st.dataframe(df.head(10))
    
    # Data Processing Section
    st.header("🔧 Data Processing")
    
    # Keep relevant columns
    df_filtered = df[['X','Y','xG','player','situation','shotType','GOAL','match_id','team ']]
    
    # Calculate normalized shots by team
    st.markdown("#### Step 1: Calculate Shot Share")
    st.markdown("""
    **Shot Share** = Player shots / Team shots in matches they played
    
    This helps separate players who are heavily involved in their team's attack.
    """)
    
    with st.spinner("Calculating shot shares..."):
        # Calculate shot share (from your original code)
        df_with_team = df_filtered.copy()
        df_with_team['match_team_id'] = df_with_team['match_id'].astype(str) + '_' + df_with_team['team '].astype(str)
        
        shots_per_match_team = df_with_team.groupby('match_team_id').size().reset_index(name='team_shots_in_match')
        player_match_unique = df_with_team[['player', 'match_team_id']].drop_duplicates()
        player_match_unique = player_match_unique.merge(shots_per_match_team, on='match_team_id', how='left')
        
        player_normalized_shots = player_match_unique.groupby('player').agg({
            'team_shots_in_match': 'sum',
            'match_team_id': 'count'
        }).rename(columns={
            'team_shots_in_match': 'total_team_shots_all_matches',
            'match_team_id': 'matches_played'
        }).reset_index()
        
        player_shot_counts = df_with_team.groupby('player').size().reset_index(name='player_shots')
        player_normalized_shots = player_normalized_shots.merge(player_shot_counts, on='player', how='left')
        player_normalized_shots['shot_share'] = (
            player_normalized_shots['player_shots'] / player_normalized_shots['total_team_shots_all_matches']
        )
    
    # Aggregate by player
    st.markdown("#### Step 2: Aggregate Player Features")
    
    with st.spinner("Aggregating player features..."):
        df_player = df_filtered.groupby('player').agg(
            X_avg = ('X', 'mean'),
            Y_std = ('Y', 'std'),
            xG_avg = ('xG','mean'),
            xG_sum = ('xG','sum'),
            Goal_sum = ('GOAL','sum'),
            total_shots = ('situation', 'count'),
            Head_percent = ('shotType', lambda x: (x== 'Head').mean()),
            Openplay_percent = ('situation', lambda x: (x=='OpenPlay').mean()),
            DirectFreekick_percent = ('situation', lambda x: (x=='DirectFreekick').mean()),
            Corner_percent = ('situation', lambda x: (x=='FromCorner').mean()),
            DirectFreekick_goal_percent = ('GOAL', lambda x: x[(df_filtered.loc[x.index, 'situation'] == 'DirectFreekick')].sum() / x.sum() if x.sum() > 0 else 0),
            Penalty_goal_percent = ('GOAL', lambda x: x[(df_filtered.loc[x.index, 'situation'] == 'Penalty')].sum() / x.sum() if x.sum() > 0 else 0),
            Openplay_goal_percent = ('GOAL', lambda x: x[(df_filtered.loc[x.index, 'situation'] == 'OpenPlay')].sum() / x.sum() if x.sum() > 0 else 0),
        ).reset_index()
        
        # Merge shot share
        df_player = df_player.merge(
            player_normalized_shots[['player', 'total_team_shots_all_matches', 'shot_share']], 
            on='player', 
            how='left'
        )
        
        # Filter by minimum shots
        players_before = len(df_player)
        df_player = df_player[df_player['total_shots'] >= MIN_SHOTS_THRESHOLD]
        players_after = len(df_player)
        
        # Calculate xG overperformance
        df_player['avgxGoverperformance'] = (df_player.Goal_sum - df_player.xG_sum) / df_player.total_shots
    
    # Show filtering results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Players Before Filtering", players_before)
    with col2:
        st.metric("Players After Filtering", players_after, delta=f"-{players_before - players_after}")
    
    # Feature Selection
    st.header("🎯 Feature Selection")
    
    cluster_features = df_player.drop(columns=['total_shots','xG_sum','Goal_sum','total_team_shots_all_matches'])
    
    # Correlation matrix
    st.markdown("#### Correlation Matrix")
    
    with st.spinner("Calculating correlations..."):
        correlation_matrix = cluster_features.drop('player', axis=1).corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=False,
                    linewidths=0.5,
                    ax=ax)
        ax.set_title('Correlation Matrix of Player Features')
        st.pyplot(fig)
    
    st.markdown("""
    **Interpretation:**
    - High correlation between Head_percent and Corner_percent (expected)
    - xG_avg correlates with X_avg (closer to goal = higher xG)
    - Low correlations indicate independent style dimensions
    """)
    
    # Clustering Section
    st.header("🎪 Player Clustering")
    
    # Prepare data for clustering
    scaler = StandardScaler()
    normalizer = Normalizer(norm='l2')
    
    cluster_features_indexed = cluster_features.set_index('player')
    X_scaled = scaler.fit_transform(cluster_features_indexed)
    X_cosine = normalizer.fit_transform(X_scaled)
    
    # Elbow Method
    st.markdown("#### Determining Optimal Number of Clusters")
    
    with st.spinner("Running elbow method..."):
        inertia_values = []
        k_range = range(2, 13)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_cosine)
            inertia_values.append(kmeans.inertia_)
        
        # Plot elbow curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, inertia_values, marker='o', color='b', linewidth=2, markersize=8)
        ax.set_title('Elbow Method for Optimal K', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        ax.set_xticks(k_range)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.markdown(f"""
    **Selected K = 5**
    
    Reasoning from analysis:
    - K=5 Messi and Ronaldo are in different clusters
    - K=6 Average radar charts are too similar between certain groups. Makes them look like the same style of player just a little bit worse.
    - K=5 Definite Distinction between player styles.
    """)
    
    # Apply clustering
    with st.spinner(f"Clustering players into {CHOSEN_K} groups..."):
        kmeans = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=10)
        df_player['cluster'] = kmeans.fit_predict(X_cosine)

    # ========== RESULTS FIRST: ENHANCED VISUALIZATIONS ==========
    st.header("🎯 Results Overview")
    st.markdown("### Player Style Clusters - Key Findings")
    
    # Define color palette
    cluster_colors_list = px.colors.qualitative.Set2[:CHOSEN_K]
    cluster_colors_dict = {i: cluster_colors_list[i] for i in range(CHOSEN_K)}
    
    # ========== t-SNE PLOT WITH FAMOUS PLAYER LABELS ==========
    st.markdown("#### Cluster Map: Player Similarity Space")
    
    # List of famous players to label
    famous_players = [
        'Erling Haaland', 'Neymar', 'Lionel Messi', 'Cristiano Ronaldo', 
        'Robert Lewandowski', 'Kylian Mbappé', 'Sergio Ramos',
        'Karim Benzema', 'Luis Suárez', 'Harry Kane', 'Mohamed Salah',
        'Zlatan Ibrahimovic', 'Gareth Bale', 'Thomas Müller', 'Eden Hazard'
    ]
    
    # Function to find cluster center representatives
    def find_cluster_representative(df, cluster_num, min_shots=100):
        """Find the most representative player for a cluster: 
        near center and famous (high shots)"""
        cluster_df = df[df['cluster'] == cluster_num].copy()
        
        # Calculate distance from cluster center
        center_x = cluster_df['tsne_x'].mean()
        center_y = cluster_df['tsne_y'].mean()
        cluster_df['dist_from_center'] = np.sqrt(
            (cluster_df['tsne_x'] - center_x)**2 + 
            (cluster_df['tsne_y'] - center_y)**2
        )
        
        # Filter for relatively famous players (high shots)
        famous_in_cluster = cluster_df[cluster_df['total_shots'] >= min_shots]
        
        if len(famous_in_cluster) == 0:
            # Fall back to any player if no one is famous enough
            famous_in_cluster = cluster_df
        
        # Find player closest to center among famous players
        representative = famous_in_cluster.nsmallest(1, 'dist_from_center')
        
        return representative['player'].values[0] if len(representative) > 0 else None
    
    # Get players to label
    players_to_label = set()
    
    # Add famous players that exist in dataset
    for player in famous_players:
        if player in df_player['player'].values:
            players_to_label.add(player)
    
    # Add cluster representatives
    for cluster_num in range(CHOSEN_K):
        rep = find_cluster_representative(df_player, cluster_num, min_shots=80)
        if rep:
            players_to_label.add(rep)
    
    # Create t-SNE coordinates
    tsne_viz = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne_viz = tsne_viz.fit_transform(X_cosine)
    df_player['tsne_x'] = X_tsne_viz[:, 0]
    df_player['tsne_y'] = X_tsne_viz[:, 1]
    
    # Create t-SNE plot with Plotly
    fig_tsne = go.Figure()
    
    # Add points for each cluster
    for cluster_num in range(CHOSEN_K):
        cluster_df = df_player[df_player['cluster'] == cluster_num]
        
        # Separate labeled and unlabeled players
        labeled_mask = cluster_df['player'].isin(players_to_label)
        
        # Unlabeled players (regular points)
        unlabeled_df = cluster_df[~labeled_mask]
        if len(unlabeled_df) > 0:
            fig_tsne.add_trace(go.Scatter(
                x=unlabeled_df['tsne_x'],
                y=unlabeled_df['tsne_y'],
                mode='markers',
                name=f'Cluster {cluster_num}',
                marker=dict(
                    size=6,
                    color=cluster_colors_dict[cluster_num],
                    opacity=0.5,
                    line=dict(width=0.5, color='white')
                ),
                text=unlabeled_df['player'],
                hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_num) + '<extra></extra>',
                showlegend=True
            ))
        
        # Labeled players (with text)
        labeled_df = cluster_df[labeled_mask]
        if len(labeled_df) > 0:
            fig_tsne.add_trace(go.Scatter(
                x=labeled_df['tsne_x'],
                y=labeled_df['tsne_y'],
                mode='markers+text',
                name=f'Cluster {cluster_num} (labeled)',
                marker=dict(
                    size=10,
                    color=cluster_colors_dict[cluster_num],
                    symbol='star',
                    opacity=0.9,
                    line=dict(width=1.5, color='black')
                ),
                text=labeled_df['player'],
                textposition='top center',
                textfont=dict(size=9, color='black'),
                hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_num) + '<extra></extra>',
                showlegend=False
            ))
    
    fig_tsne.update_layout(
        title=dict(
            text='<b>Player Style Clusters - t-SNE Projection</b><br><sub>Famous players and cluster representatives labeled</sub>',
            font=dict(size=18)
        ),
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        height=700,
        hovermode='closest',
        legend=dict(
            title='Clusters',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig_tsne, use_container_width=True)
    
    st.markdown("""
    **Understanding the Map:**
    - Each point is a player, colored by their cluster
    - ⭐ Stars indicate famous players or cluster representatives
    - Players close together have similar playing styles
    - Distinct clusters show different player archetypes
    """)
    
    # ========== CLUSTER RADAR CHARTS ==========
    st.markdown("#### Cluster Style Profiles")
    st.markdown("Average characteristics of each playing style:")
    
    # Define features for radar charts
    radar_features_viz = [
        'xG_avg', 'X_avg', 'Y_std', 'Head_percent',
        'Openplay_percent', 'shot_share',
        'avgxGoverperformance', 'DirectFreekick_percent'
    ]
    
    feature_labels_viz = [
        'Chance\nQuality', 'Goal\nProximity', 'Movement\nRange',
        'Headers', 'Open\nPlay', 'Talisman', 'Clinical', 'Set Piece\nTaker'
    ]
    
    # Normalize features for radar chart
    def normalize_for_radar(df_column):
        """Normalize values to 0-1 scale for radar chart"""
        min_val = df_column.min()
        max_val = df_column.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(df_column), index=df_column.index)
        return (df_column - min_val) / (max_val - min_val)
    
    # Create normalized dataframe
    radar_normalized_viz = pd.DataFrame()
    for feature in radar_features_viz:
        if feature in df_player.columns:
            radar_normalized_viz[feature] = normalize_for_radar(df_player[feature])
        else:
            radar_normalized_viz[feature] = 0.5
    
    radar_normalized_viz['cluster'] = df_player['cluster'].values
    
    # Calculate cluster averages
    cluster_averages_viz = []
    for cluster_num in range(CHOSEN_K):
        cluster_mask = radar_normalized_viz['cluster'] == cluster_num
        cluster_size = sum(cluster_mask)
        
        if cluster_size > 0:
            avg_values = []
            for feature in radar_features_viz:
                cluster_feature_vals = radar_normalized_viz[cluster_mask][feature]
                avg_values.append(cluster_feature_vals.mean())
            
            # Get representative player for title
            rep_player = find_cluster_representative(df_player, cluster_num, min_shots=80)
            
            cluster_averages_viz.append({
                'cluster': cluster_num,
                'size': cluster_size,
                'values': avg_values,
                'color': cluster_colors_dict[cluster_num],
                'representative': rep_player
            })
    
    # Display radar charts in grid
    cols_per_row = 3
    rows_needed = (CHOSEN_K + cols_per_row - 1) // cols_per_row
    
    for row in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            cluster_idx = row * cols_per_row + col_idx
            if cluster_idx < len(cluster_averages_viz):
                with cols[col_idx]:
                    cluster_data = cluster_averages_viz[cluster_idx]
                    
                    # Create radar chart
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=cluster_data['values'] + cluster_data['values'][:1],
                        theta=feature_labels_viz + feature_labels_viz[:1],
                        fill='toself',
                        fillcolor=cluster_data['color'],
                        line_color=cluster_data['color'],
                        opacity=0.6,
                        name=f'Cluster {cluster_data["cluster"]}'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                showticklabels=False
                            )
                        ),
                        title=dict(
                            text=f"<b>Cluster {cluster_data['cluster']}</b><br><sub>{cluster_data['representative']}</sub><br><sub style='color:gray'>{cluster_data['size']} players</sub>",
                            font=dict(size=13)
                        ),
                        showlegend=False,
                        height=350,
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
    
    with st.expander("📊 How to Read Cluster Radar Charts"):
        st.markdown("""
        **Feature Explanations:**
        - **Chance Quality**: Average xG per shot (higher = better chances)
        - **Goal Proximity**: Average X coordinate (higher = closer to goal)
        - **Movement Range**: Standard deviation of Y position (higher = wider movement)
        - **Headers**: Percentage of shots with head (higher = more aerial threat)
        - **Open Play**: Percentage of shots from open play (higher = less set-piece dependent)
        - **Talisman**: Share of team's shots (higher = more central to attack)
        - **Clinical**: xG overperformance (higher = better finishing than expected)
        - **Set Piece Taker**: Percentage of direct free kicks taken
        
        **Interpretation Tips:**
        1. Compare cluster shapes to identify style differences
        2. Larger areas indicate more well-rounded profiles
        3. Spikes show cluster specialties
        4. Similar shapes indicate similar playing styles
        """)
    
    st.markdown("---")
    
    # ========== ORIGINAL SECTIONS CONTINUE BELOW ==========
    
        
        # Show cluster distribution
    cluster_counts = df_player['cluster'].value_counts().sort_index()
        
    cols = st.columns(CHOSEN_K)
    for i, col in enumerate(cols):
            if i < len(cluster_counts):
                col.metric(f"Cluster {i}", cluster_counts.iloc[i])
    
    # Cluster Visualization
    st.header("📈 Cluster Visualizations")
    
    # PCA Visualization
    st.markdown("#### Principal Component Analysis (PCA)")
    
    with st.spinner("Creating PCA visualization..."):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_cosine)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors for clusters
        colors = plt.cm.viridis(np.linspace(0, 1, CHOSEN_K))
        
        for i in range(CHOSEN_K):
            mask = (df_player['cluster'] == i)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f'Cluster {i}', 
                      color=colors[i],
                      alpha=0.6,
                      s=30)
        
        ax.set_title(f'Player Clusters ({CHOSEN_K} clusters) - PCA Visualization', fontsize=16)
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.legend(title='Cluster')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # PCA Interpretation
    st.markdown("#### PCA Component Analysis")
    
    feature_names = cluster_features.drop('player', axis=1).columns
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top contributors to PC1:**")
        top_pc1 = pca_components['PC1'].abs().sort_values(ascending=False).head(5)
        st.dataframe(top_pc1)
    
    with col2:
        st.markdown("**Top contributors to PC2:**")
        top_pc2 = pca_components['PC2'].abs().sort_values(ascending=False).head(5)
        st.dataframe(top_pc2)
    
    st.markdown("""
    **Interpretation:**
    - **PC1 (Horizontal):** Separates by heading ability, width, and set-piece involvement
    - **PC2 (Vertical):** Separates by chance quality, proximity to goal, and shot share
    """)
    
    # t-SNE Visualization
    st.markdown("#### t-SNE Visualization")
    
    with st.spinner("Creating t-SNE visualization..."):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_cosine)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                            c=df_player['cluster'], 
                            cmap='viridis', 
                            alpha=0.6,
                            s=30)
        ax.set_title('Player Clusters - t-SNE Visualization')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
    st.markdown("""
    **t-SNE Insight:**
    - Shows natural groupings based on style similarity
    - Clear separation between clusters
    - Helps validate cluster quality
    """)
     # ============================================================================
    # NEW: CLUSTER AVERAGE RADAR CHARTS
    # ============================================================================
    st.markdown("#### Cluster Average Style Profiles")
    st.markdown("Radar charts showing the average style profile for each cluster:")
    
    # Define features for radar charts
    radar_features = [
        'xG_avg', 'X_avg', 'Y_std', 'Head_percent',
        'Openplay_percent', 'shot_share',
        'avgxGoverperformance', 'DirectFreekick_percent'
    ]
    
    feature_labels = [
        'Chance\nQuality', 'Goal\nProximity', 'Movement\nRange',
        'Headers', 'Open\nPlay', 'Talisman', 'Clinical', 'Set Piece\nTaker'
    ]
    
    # Create normalized versions for radar chart display
    def normalize_for_radar(df_column):
        """Normalize values to 0-1 scale for radar chart"""
        min_val = df_column.min()
        max_val = df_column.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(df_column), index=df_column.index)
        return (df_column - min_val) / (max_val - min_val)
    
    # Create normalized dataframe for radar charts
    radar_normalized = pd.DataFrame()
    for feature in radar_features:
        if feature in df_player.columns:
            radar_normalized[feature] = normalize_for_radar(df_player[feature])
        else:
            radar_normalized[feature] = 0.5
    
    # Calculate cluster averages
    cluster_averages = []
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, CHOSEN_K))
    
    for cluster_num in range(CHOSEN_K):
        cluster_mask = df_player['cluster'] == cluster_num
        cluster_size = sum(cluster_mask)
        
        if cluster_size > 0:
            # Get average values for this cluster
            avg_values = []
            for feature in radar_features:
                if feature in df_player.columns:
                    # Get average of normalized values
                    cluster_feature_vals = radar_normalized[cluster_mask][feature]
                    avg_values.append(cluster_feature_vals.mean())
                else:
                    avg_values.append(0.5)
            
            cluster_averages.append({
                'cluster': cluster_num,
                'size': cluster_size,
                'values': avg_values,
                'color': cluster_colors[cluster_num]
            })
    
    # Create radar charts in columns
    cols_per_row = 3
    rows_needed = (CHOSEN_K + cols_per_row - 1) // cols_per_row
    
    for row in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            cluster_idx = row * cols_per_row + col_idx
            if cluster_idx < len(cluster_averages):
                with cols[col_idx]:
                    cluster_data = cluster_averages[cluster_idx]
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    # Add radar trace
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_data['values'] + cluster_data['values'][:1],  # Close the polygon
                        theta=feature_labels + feature_labels[:1],
                        fill='toself',
                        fillcolor=f'rgba({int(cluster_data["color"][0]*255)},{int(cluster_data["color"][1]*255)},{int(cluster_data["color"][2]*255)},0.3)',
                        line_color=f'rgb({int(cluster_data["color"][0]*255)},{int(cluster_data["color"][1]*255)},{int(cluster_data["color"][2]*255)})',
                        name=f'Cluster {cluster_data["cluster"]}'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                tickvals=[0, 0.25, 0.5, 0.75, 1],
                                ticktext=["0%", "25%", "50%", "75%", "100%"]
                            ),
                            angularaxis=dict(
                                direction="clockwise"
                            )
                        ),
                        title=f"Cluster {cluster_data['cluster']}<br><span style='font-size:12px'>{cluster_data['size']} players</span>",
                        showlegend=False,
                        height=350,
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    with st.expander("📊 How to Read Cluster Radar Charts"):
        st.markdown("""
        **Feature Explanations:**
        - **Chance Quality**: Average xG per shot (higher = better chances)
        - **Goal Proximity**: Average X coordinate (higher = closer to goal)
        - **Movement Range**: Standard deviation of Y position (higher = wider movement)
        - **Headers**: Percentage of shots with head (higher = more aerial threat)
        - **Open Play**: Percentage of shots from open play (higher = less set-piece dependent)
        - **Talisman**: Share of team's shots (higher = more central to attack)
        - **Clinical**: xG overperformance (higher = better finishing than expected)
        - **Set Piece Taker**: Percentage of direct free kicks taken
        
        **Interpretation Tips:**
        1. Compare cluster shapes to identify style differences
        2. Larger areas indicate more well-rounded profiles
        3. Spikes show cluster specialties
        4. Similar shapes indicate similar playing styles
        """)
    # Cluster Analysis
    st.header("🔍 Cluster Analysis")
    
    # Show top players in each cluster
    st.markdown("#### Top Players in Each Cluster")
    
    for cluster_num in sorted(df_player['cluster'].unique()):
        with st.expander(f"Cluster {cluster_num} - {len(df_player[df_player['cluster'] == cluster_num])} players"):
            cluster_df = df_player[df_player['cluster'] == cluster_num]
            top_players = cluster_df.sort_values('total_shots', ascending=False).head(10)
            
            # Show as table
            st.dataframe(top_players[['player', 'total_shots', 'xG_avg', 'shot_share', 
                                      'Openplay_percent', 'Head_percent']])
    
    # Radar Charts
    st.header("📊 Player Style Profiles")
    
    radar_features = [
        'xG_avg', 'X_avg', 'Y_std', 'Head_percent',
        'Openplay_percent', 'shot_share',
        'avgxGoverperformance', 'DirectFreekick_percent'
    ]
    
    feature_labels = [
        'Chance\nQuality', 'Proximity\nTo Goal', 'Movement\nRange',
        'Takes\nHeaders', 'Open\nPlay\nParticipation', 'Talisman', 'Clinical', 'Set Piece\n Taker'
    ]
    
    # Create radar chart for a selected player
    st.markdown("#### Compare Player Styles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox("Select Player 1", 
                              df_player['player'].sort_values().tolist(),
                              index=df_player['player'].tolist().index('Lionel Messi') if 'Lionel Messi' in df_player['player'].tolist() else 0)
    
    with col2:
        player2 = st.selectbox("Select Player 2", 
                              df_player['player'].sort_values().tolist(),
                              index=df_player['player'].tolist().index('Cristiano Ronaldo') if 'Cristiano Ronaldo' in df_player['player'].tolist() else 1)
    
    if player1 and player2:
        # Normalize for radar chart
        def normalize_feature(feature_series):
            min_val = feature_series.min()
            max_val = feature_series.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(feature_series), index=feature_series.index)
            return (feature_series - min_val) / (max_val - min_val)
        
        # Get normalized values
        radar_data = pd.DataFrame()
        for feature in radar_features:
            if feature in df_player.columns:
                radar_data[feature] = normalize_feature(df_player[feature])
        
        # Get player indices
        idx1 = df_player[df_player['player'] == player1].index[0]
        idx2 = df_player[df_player['player'] == player2].index[0]
        
        # Prepare data for radar chart
        categories = feature_labels
        values1 = [radar_data.loc[idx1, feature] for feature in radar_features]
        values2 = [radar_data.loc[idx2, feature] for feature in radar_features]
        
        # Create radar chart using Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values1 + values1[:1],  # Close the polygon
            theta=categories + categories[:1],
            fill='toself',
            name=player1,
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values2 + values2[:1],
            theta=categories + categories[:1],
            fill='toself',
            name=player2,
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"Style Comparison: {player1} vs {player2}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Player Finder Tool
    st.header("🔎 Player Similarity Finder")
    
    st.markdown("Find players with similar styles to your target player:")
    
    target_player = st.selectbox("Select Target Player", 
                                df_player['player'].sort_values().tolist())
    
    if target_player:
        # Get target player's cluster
        target_cluster = df_player[df_player['player'] == target_player]['cluster'].values[0]
        
        # Get similar players (same cluster, excluding target)
        similar_players = df_player[
            (df_player['cluster'] == target_cluster) & 
            (df_player['player'] != target_player)
        ].copy()
        
        # Sort by similarity (Euclidean distance in feature space)
        target_features = cluster_features_indexed.loc[target_player].values
        
        def calculate_distance(player_row):
            player_features = cluster_features_indexed.loc[player_row['player']].values
            return np.linalg.norm(target_features - player_features)
        
        similar_players['distance'] = similar_players.apply(calculate_distance, axis=1)
        similar_players = similar_players.sort_values('distance').head(10)
        
        st.markdown(f"**Players with similar style to {target_player} (Cluster {target_cluster}):**")
        
        # Display similar players
        cols = st.columns(5)
        for idx, (_, row) in enumerate(similar_players.iterrows()):
            if idx < 10:
                with cols[idx % 5]:
                    st.metric(
                        row['player'],
                        f"{row['total_shots']} shots",
                        delta=f"Dist: {row['distance']:.3f}"
                    )
    
    # Methodology Explanation
    st.header("📖 Methodology Deep Dive")
    
    with st.expander("View Detailed Methodology"):
        st.markdown("""
        ### Project Goal
        Create a tool to find new players similar in style to known players using unsupervised machine learning.
        
        ### Why Cosine K-Means?
        - Player style is akin to a vector direction
        - Cosine similarity compares direction, not magnitude
        - Eliminates volume bias (old players with many shots vs new players with few)
        - Focuses on style rather than output
        
        ### Data Processing Steps:
        1. **Data Loading**: 400k shots from 2014-2022 in European leagues
        2. **Feature Engineering**:
           - Average position (X_avg, Y_std)
           - Situation percentages (OpenPlay, Corner, etc.)
           - Shot type percentages (Head, Foot)
           - xG metrics (average, sum, overperformance)
           - Shot share (% of team shots)
        3. **Filtering**: Minimum 40 shots to reduce noise
        4. **Feature Selection**: Manual selection using correlation matrix
        5. **Scaling**: StandardScaler followed by L2 normalization
        6. **Clustering**: Cosine K-Means with K=6 (optimal separation)
        
        ### Key Insights:
        - **K=6** provides optimal separation
        - Defensive players cluster separately (mainly participate in set pieces)
        - Messi and Ronaldo are in different clusters (different styles)
        - Clear separation between open-play specialists and set-piece specialists
        """)
    
    # Download Results
    st.header("💾 Download Results")
    
    # Prepare download data
    csv = df_player.to_csv(index=False)
    st.download_button(
        label="Download Player Clusters (CSV)",
        data=csv,
        file_name="player_clusters.csv",
        mime="text/csv",
    )

else:
    st.error("Could not load data. Please ensure 'datacompleta.parquet' is in the same directory.")
    st.info("""
    For this app to work, you need:
    1. The data file 'datacompleta.parquet' in your repository
    2. Required packages: streamlit, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn
    
    To run locally: `streamlit run app.py`
    """)

# Footer
st.markdown("---")
st.markdown("""
**Data Science Portfolio Project** | Created with Python, Scikit-learn, and Streamlit
""")
