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
    # Data Processing (silently in background)
    with st.spinner("Processing data..."):
        # Keep relevant columns
        df_filtered = df[['X','Y','xG','player','situation','shotType','GOAL','match_id','team ']]
        
        # Calculate normalized shots by team
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
        df_player = df_player[df_player['total_shots'] >= MIN_SHOTS_THRESHOLD]
        
        # Calculate xG overperformance
        df_player['avgxGoverperformance'] = (df_player.Goal_sum - df_player.xG_sum) / df_player.total_shots
        
        # Feature Selection
        cluster_features = df_player.drop(columns=['total_shots','xG_sum','Goal_sum','total_team_shots_all_matches'])
        cluster_features_indexed = cluster_features.set_index('player')
        
        # Scaling and Clustering
        scaler = StandardScaler()
        cluster_features_scaled = scaler.fit_transform(cluster_features_indexed)
        
        normalizer = Normalizer(norm='l2')
        cluster_features_normalized = normalizer.fit_transform(cluster_features_scaled)
        
        # Perform clustering
        kmeans_cosine = KMeans(n_clusters=CHOSEN_K, init='k-means++', n_init=50, max_iter=500, random_state=42)
        df_player['cluster'] = kmeans_cosine.fit_predict(cluster_features_normalized)
        
        # t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(cluster_features_normalized)
        df_player['tsne_1'] = tsne_results[:, 0]
        df_player['tsne_2'] = tsne_results[:, 1]
    
    # Define color palette for clusters
    color_palette = px.colors.qualitative.Set2[:CHOSEN_K]
    cluster_colors = {i: color_palette[i] for i in range(CHOSEN_K)}
    
    # Function to find most famous player in cluster (based on total shots)
    def get_top_player_by_cluster(df, cluster_num):
        cluster_df = df[df['cluster'] == cluster_num]
        return cluster_df.sort_values('total_shots', ascending=False).iloc[0]
    
    # ========== TOP SECTION: RADAR CHARTS AND T-SNE ==========
    st.header("🎯 Cluster Profiles Overview")
    
    # Radar chart features
    radar_features = [
        'xG_avg', 'X_avg', 'Y_std', 'Head_percent',
        'Openplay_percent', 'shot_share',
        'avgxGoverperformance', 'DirectFreekick_percent'
    ]
    
    feature_labels = [
        'Chance\nQuality', 'Proximity\nTo Goal', 'Movement\nRange',
        'Takes\nHeaders', 'Open\nPlay', 'Talisman', 'Clinical', 'Set Piece\nTaker'
    ]
    
    # Normalize for radar chart
    def normalize_feature(feature_series):
        min_val = feature_series.min()
        max_val = feature_series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(feature_series), index=feature_series.index)
        return (feature_series - min_val) / (max_val - min_val)
    
    # Get normalized values for all players
    radar_data = pd.DataFrame()
    for feature in radar_features:
        if feature in df_player.columns:
            radar_data[feature] = normalize_feature(df_player[feature])
    radar_data.index = df_player.index
    
    # Create radar charts for each cluster (top player)
    st.markdown("### Representative Players by Cluster")
    
    # Determine layout based on number of clusters
    if CHOSEN_K <= 3:
        cols = st.columns(CHOSEN_K)
    elif CHOSEN_K <= 6:
        cols = st.columns(3)
    else:
        cols = st.columns(4)
    
    for i in range(CHOSEN_K):
        col_idx = i % len(cols)
        with cols[col_idx]:
            top_player_row = get_top_player_by_cluster(df_player, i)
            player_name = top_player_row['player']
            
            # Get player index in df_player
            player_idx = df_player[df_player['player'] == player_name].index[0]
            
            # Get radar values
            radar_values = [radar_data.loc[player_idx, feature] for feature in radar_features]
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=radar_values + radar_values[:1],
                theta=feature_labels + feature_labels[:1],
                fill='toself',
                name=player_name,
                line_color=cluster_colors[i],
                fillcolor=cluster_colors[i],
                opacity=0.6
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        showticklabels=False
                    )
                ),
                showlegend=False,
                title=dict(
                    text=f"<b>{player_name}</b><br>Cluster {i}",
                    font=dict(size=14)
                ),
                height=300,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # t-SNE Cluster Plot
    st.markdown("### t-SNE Cluster Visualization")
    
    # Get top player for each cluster for labeling
    top_players_per_cluster = {}
    for cluster_num in range(CHOSEN_K):
        top_player = get_top_player_by_cluster(df_player, cluster_num)
        top_players_per_cluster[cluster_num] = top_player['player']
    
    # Create t-SNE plot
    fig = go.Figure()
    
    for cluster_num in range(CHOSEN_K):
        cluster_data = df_player[df_player['cluster'] == cluster_num]
        
        # Add scatter points for cluster
        fig.add_trace(go.Scatter(
            x=cluster_data['tsne_1'],
            y=cluster_data['tsne_2'],
            mode='markers',
            name=f'Cluster {cluster_num}',
            marker=dict(
                size=8,
                color=cluster_colors[cluster_num],
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            text=cluster_data['player'],
            hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_num) + '<extra></extra>'
        ))
        
        # Add label for top player
        top_player_name = top_players_per_cluster[cluster_num]
        top_player_data = cluster_data[cluster_data['player'] == top_player_name]
        
        fig.add_trace(go.Scatter(
            x=top_player_data['tsne_1'],
            y=top_player_data['tsne_2'],
            mode='markers+text',
            name=top_player_name,
            marker=dict(
                size=12,
                color=cluster_colors[cluster_num],
                symbol='star',
                line=dict(width=2, color='black')
            ),
            text=[top_player_name],
            textposition='top center',
            textfont=dict(size=10, color='black'),
            showlegend=False,
            hovertemplate='<b>%{text}</b><br>Top Player - Cluster: ' + str(cluster_num) + '<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>Player Style Clusters (t-SNE Projection)</b>',
            font=dict(size=18)
        ),
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        height=600,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Understanding the Visualization:**
    - Each point represents a player
    - Colors indicate cluster membership
    - ⭐ Stars mark the most prominent player in each cluster
    - Similar styles cluster together in 2D space
    """)
    
    st.markdown("---")
    
    # ========== REST OF THE ORIGINAL APP CONTINUES BELOW ==========
    
    # Main content
    st.header("📊 Data Overview")
    
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
    
    # Show filtering results
    st.markdown("#### Step 1: Calculate Shot Share")
    st.markdown("""
    **Shot Share** = Player shots / Team shots in matches they played
    
    This helps separate players who are heavily involved in their team's attack.
    """)
    
    players_before = df['player'].nunique()
    players_after = len(df_player)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Players Before Filtering", players_before)
    with col2:
        st.metric("Players After Filtering", players_after, delta=f"-{players_before - players_after}")
    
    # Feature Selection
    st.header("🎯 Feature Selection")
    
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
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("""
    **Interpretation:**
    - Values close to +1 indicate strong positive correlation
    - Values close to -1 indicate strong negative correlation
    - Values close to 0 indicate little to no correlation
    """)
    
    # Clustering Section
    st.header("🎲 K-Means Clustering")
    
    st.markdown(f"**Using K={CHOSEN_K} clusters with Cosine K-Means**")
    
    # Show cluster distribution
    st.markdown("#### Cluster Distribution")
    
    cluster_counts = df_player['cluster'].value_counts().sort_index()
    
    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Number of Players'},
        title='Players per Cluster',
        color=cluster_counts.index,
        color_discrete_map=cluster_colors
    )
    
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart explanation
    st.markdown("""
    #### Radar Chart Interpretation
    
    **Features Explained:**
    - **Chance Quality**: Average xG per shot (higher = better quality chances)
    - **Proximity to Goal**: Average X position (higher = closer to goal)
    - **Movement Range**: Standard deviation of Y position (higher = more lateral movement)
    - **Takes Headers**: Percentage of shots that are headers
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
    
    # Radar Charts - Interactive Player Comparison
    st.header("📊 Player Style Profiles")
    
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
        # Get player indices
        idx1 = df_player[df_player['player'] == player1].index[0]
        idx2 = df_player[df_player['player'] == player2].index[0]
        
        # Get clusters for color coding
        cluster1 = df_player.loc[idx1, 'cluster']
        cluster2 = df_player.loc[idx2, 'cluster']
        
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
            name=f"{player1} (Cluster {cluster1})",
            line_color=cluster_colors[cluster1]
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values2 + values2[:1],
            theta=categories + categories[:1],
            fill='toself',
            name=f"{player2} (Cluster {cluster2})",
            line_color=cluster_colors[cluster2]
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
