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

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Player Style Clustering Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# GLOBAL CLUSTER COLOR MAPPING (Consistent across all visualizations)
# =============================================================================
# Define colors for clusters 0-12 (supports K up to 13)
CLUSTER_COLOR_MAP = {
    0: '#FEE440',   # Yellow
    1: '#F15BB5',   # Magenta
    2: '#00B4D8',   # Cyan
    3: '#E63946',   # Red
    4: '#003566',   # Deep Blue
    5: '#00F5D4',   # Bright Teal
    6: '#9B5DE5',   # Violet
    7: '#2D3142',   # Slate Charcoal
    8: '#D9D9D9',   # Gray
    9: '#BC80BD',   # Purple
    10: '#CCEBC5',  # Mint
    11: '#FFED6F',  # Yellow
    12: '#E78AC3'   # Rose
}

# Helper function to get consistent colors
def get_cluster_color(cluster_num):
    """Get consistent color for a cluster number"""
    return CLUSTER_COLOR_MAP.get(cluster_num, '#8DD3C7')  # Default to first color if out of range

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CACHED DATA LOADING AND PROCESSING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_raw_data():
    """Load raw data from parquet file - cached across sessions"""
    try:
        df = pd.read_parquet('datacompleta.parquet')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(show_spinner=False)
def preprocess_data(df, min_shots_threshold):
    """
    Complete data preprocessing pipeline - cached by threshold
    Returns processed player dataframe ready for clustering
    """
    # Filter relevant columns
    df_filtered = df[['X','Y','xG','player','situation','shotType','GOAL','match_id','team ']]
    
    # Calculate shot share
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
    
    # Aggregate player statistics
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
    df_player = df_player[df_player['total_shots'] >= min_shots_threshold]
    
    # Calculate xG overperformance
    df_player['avgxGoverperformance'] = (df_player.Goal_sum - df_player.xG_sum) / df_player.total_shots
    
    return df_player

@st.cache_data(show_spinner=False)
def perform_clustering(df_player, n_clusters):
    """
    Perform clustering and dimensionality reduction - cached by K
    Returns df_player with cluster assignments and coordinates
    """
    # Prepare features
    cluster_features = df_player.drop(columns=['total_shots','xG_sum','Goal_sum','total_team_shots_all_matches'])
    cluster_features_indexed = cluster_features.set_index('player')
    
    # Scale and normalize
    scaler = StandardScaler()
    normalizer = Normalizer(norm='l2')
    
    X_scaled = scaler.fit_transform(cluster_features_indexed)
    X_cosine = normalizer.fit_transform(X_scaled)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cosine)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_cosine)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cosine)
    
    # Add to dataframe
    df_result = df_player.copy()
    df_result['cluster'] = clusters
    df_result['tsne_x'] = X_tsne[:, 0]
    df_result['tsne_y'] = X_tsne[:, 1]
    df_result['pca_1'] = X_pca[:, 0]
    df_result['pca_2'] = X_pca[:, 1]
    
    return df_result, X_cosine, cluster_features_indexed, pca

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_cluster_representative(df, cluster_num, min_shots=100):
    """Find most representative player for a cluster"""
    cluster_df = df[df['cluster'] == cluster_num].copy()
    
    # Calculate distance from cluster center
    center_x = cluster_df['X_avg'].mean()
    center_y = cluster_df['Y_std'].mean()
    cluster_df['dist_from_center'] = np.sqrt(
        (cluster_df['X_avg'] - center_x)**2 + 
        (cluster_df['Y_std'] - center_y)**2
    )
    
    # Filter for famous players
    famous_in_cluster = cluster_df[cluster_df['total_shots'] >= min_shots]
    if len(famous_in_cluster) == 0:
        famous_in_cluster = cluster_df
    
    # Find closest to center
    representative = famous_in_cluster.nsmallest(1, 'dist_from_center')
    return representative['player'].values[0] if len(representative) > 0 else None

def normalize_for_radar(df_column):
    """Normalize values to 0-1 scale for radar charts"""
    min_val = df_column.min()
    max_val = df_column.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(df_column), index=df_column.index)
    return (df_column - min_val) / (max_val - min_val)

# =============================================================================
# MAIN APP
# =============================================================================

# Title and sidebar
st.title("⚽ Player Style Clustering Analysis")
st.markdown("### Unsupervised Machine Learning for Football Player Similarity")

# Sidebar controls
st.sidebar.header("Analysis Controls")
MIN_SHOTS_THRESHOLD = st.sidebar.slider("Minimum Shots Threshold", min_value=20, max_value=100, value=40)
CHOSEN_K = st.sidebar.selectbox("Number of Clusters (K)", options=[4, 5, 6, 7, 8], index=1)

st.sidebar.markdown("---")
st.sidebar.header("📚 Methodology")
st.sidebar.markdown("""
**Cosine K-Means Clustering** groups players by style rather than volume.

**Key Features:**
- Shot share (% of team shots)
- Average position (X, Y)
- Situation percentages
- Shot type percentages
- xG metrics
""")

# Load and process data (all cached!)
with st.spinner("Loading data..."):
    df_raw = load_raw_data()

if df_raw is None:
    st.error("Could not load data. Please ensure 'datacompleta.parquet' is available.")
    st.stop()

# Process data (cached by threshold)
with st.spinner("Processing player statistics..."):
    df_player = preprocess_data(df_raw, MIN_SHOTS_THRESHOLD)

# Perform clustering (cached by K)
with st.spinner("Clustering players..."):
    df_player, X_cosine, cluster_features_indexed, pca = perform_clustering(df_player, CHOSEN_K)

# =============================================================================
# RESULTS FIRST - THE MONEY SHOT
# =============================================================================

st.header("🎯 Results Overview")
st.markdown("### Player Style Clusters - Key Findings")

# Color palette
# Use consistent cluster colors
cluster_colors = {i: get_cluster_color(i) for i in range(CHOSEN_K)}

# Famous players to label
famous_players = [
        'Erling Haaland', 'Neymar', 'Lionel Messi', 'Cristiano Ronaldo', 'Robert Lewandowski', 'Kylian Mbappe-Lottin', 'Antoine Griezmann', 'Paulo Dybala', 'Bruno Fernandes',
        'Marcelo', 'Toni Kroos', 'Ousmane Dembélé', 'Memphis Depay', 'Son Heung-Min', 'Kevin De Bruyne', 'Sergio Ramos', 'Karim Benzema', 'Luis Suárez', 'Harry Kane', 'Mohamed Salah',
        'Zlatan Ibrahimovic', 'Gareth Bale', 'Thomas Müller', 'Eden Hazard', 'Gavi', 'Jude Bellingham', 'Eduardo Camavinga', 'Aurelien Tchouameni', 'Jamal Musiala', 'Josko Gvardiol',
        'William Saliba', 'Nuno Mendes', 'Nico Williams', 'Robert Lewandowski', 'Kylian Mbappe-Lottin', 'Antoinne Griezmann', 'Paulo Dybala', 'Bruno Fernandes', 'Marcelo', 'Toni Kroos', 
        'Ousmane Dembélé', 'Memphis Depay', 'Son Heung-Min', 'Kevin De Bruyne', 'Sergio Ramos', 'Karim Benzema', 'Luis Suárez', 'Harry Kane', 'Mohamed Salah', 'Zlatan Ibrahimovic', 
        'Gareth Bale', 'Thomas Müller', 'Eden Hazard', 'Gavi', 'Jude Bellingham', 'Eduardo Camavinga', 'Aurelien Tchouameni', 'Jamal Musiala', 'Josko Gvardiol', 'William Saliba', 'Nuno Mendes', 'Nico Williams',
        'Luka Modric', 'Ángel Di María', 'Riyad Mahrez', 'Philippe Coutinho', 'Ilkay Gündogan', 'Christian Eriksen', 'Casemiro', 'Bernardo Silva'
    ]

# Get players to label
players_to_label = set()
for player in famous_players:
    if player in df_player['player'].values:
        players_to_label.add(player)

for cluster_num in range(CHOSEN_K):
    rep = find_cluster_representative(df_player, cluster_num, min_shots=80)
    if rep:
        players_to_label.add(rep)

# ---------- SIDE-BY-SIDE: t-SNE MAP AND CLUSTER RADARS ----------
st.markdown("#### 🗺️ Interactive Cluster Analysis")

# Create two columns for side-by-side display
col_tsne, col_radar = st.columns([1.2, 1])

# Left column: t-SNE Map
with col_tsne:
    st.markdown("**Cluster Map: Player Similarity Space**")

    
    
    
    fig_tsne = go.Figure()
    
    for cluster_num in range(CHOSEN_K):
        cluster_df = df_player[df_player['cluster'] == cluster_num]
        labeled_mask = cluster_df['player'].isin(players_to_label)
        
        # Unlabeled players
        unlabeled_df = cluster_df[~labeled_mask]
        if len(unlabeled_df) > 0:
            fig_tsne.add_trace(go.Scatter(
                x=unlabeled_df['tsne_x'],
                y=unlabeled_df['tsne_y'],
                mode='markers',
                name=f'Cluster {cluster_num}',
                marker=dict(
                    size=6,
                    color=cluster_colors[cluster_num],
                    opacity=0.5,
                    line=dict(width=0.5, color='white')
                ),
                text=unlabeled_df['player'],
                customdata=unlabeled_df['player'],
                hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_num) + '<extra></extra>',
                showlegend=True
            ))
        
        # Labeled players
        labeled_df = cluster_df[labeled_mask]
        if len(labeled_df) > 0:
            fig_tsne.add_trace(go.Scatter(
                x=labeled_df['tsne_x'],
                y=labeled_df['tsne_y'],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=cluster_colors[cluster_num],
                    symbol='star',
                    opacity=0.9,
                    line=dict(width=1.5, color='black')
                ),
                text=labeled_df['player'],
                textposition='top center',
                textfont=dict(size=9, color='black'),
                customdata=labeled_df['player'],
                hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_num) + '<extra></extra>',
                showlegend=False
            ))
    
    fig_tsne.update_layout(
        title=dict(
            text='<b>Player Style Clusters</b><br><sub>Hover over players to see their radar chart →</sub>',
            font=dict(size=16)
        ),
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        height=700,
        hovermode='closest',
        legend=dict(title='Clusters', yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    st.plotly_chart(fig_tsne, use_container_width=True)
    
    st.markdown("""
    **Understanding the Map:**
    - Each point is a player, colored by their cluster
    - ⭐ Stars indicate famous players
    - Hover to compare with cluster average →
    """)

# Right column: Interactive Radar Charts
with col_radar:
    st.markdown("**Cluster Average Profiles**")
    st.markdown("*Select a player from the dropdown or hover on the map*")
    
    # Dropdown to select player for comparison
    selected_player = st.selectbox(
        "Choose a player:",
        ['-- Cluster Averages Only --'] + sorted(df_player['player'].tolist()),
        key='radar_player_select'
    )
    
    # Calculate radar features and normalize
    radar_features = [
        'xG_avg', 'X_avg', 'Y_std', 'Head_percent',
        'Openplay_percent', 'shot_share',
        'avgxGoverperformance', 'DirectFreekick_percent'
    ]
    
    feature_labels = [
        'Chance\nQuality', 'Goal\nProximity', 'Movement\nRange',
        'Headers', 'Open\nPlay', 'Talisman', 'Clinical', 'Set Piece\nTaker'
    ]
    
    # Normalize
    radar_normalized = pd.DataFrame()
    for feature in radar_features:
        if feature in df_player.columns:
            radar_normalized[feature] = normalize_for_radar(df_player[feature])
        else:
            radar_normalized[feature] = 0.5
    radar_normalized.index = df_player.index
    
    # Calculate cluster averages
    cluster_averages = {}
    for cluster_num in range(CHOSEN_K):
        cluster_mask = df_player['cluster'] == cluster_num
        cluster_size = sum(cluster_mask)
        
        if cluster_size > 0:
            avg_values = []
            for feature in radar_features:
                cluster_feature_vals = radar_normalized[cluster_mask][feature]
                avg_values.append(cluster_feature_vals.mean())
            
            cluster_averages[cluster_num] = {
                'values': avg_values,
                'size': cluster_size,
                'color': get_cluster_color(cluster_num)
            }
    
    # Create the radar chart
    fig_radar_interactive = go.Figure()
    
    if selected_player != '-- Cluster Averages Only --':
        # Show selected player vs their cluster average
        player_idx = df_player[df_player['player'] == selected_player].index[0]
        player_cluster = df_player.loc[player_idx, 'cluster']
        player_values = [radar_normalized.loc[player_idx, feature] for feature in radar_features]
        
        # Add cluster average (semi-transparent)
        cluster_avg = cluster_averages[player_cluster]
        fig_radar_interactive.add_trace(go.Scatterpolar(
            r=cluster_avg['values'] + cluster_avg['values'][:1],
            theta=feature_labels + feature_labels[:1],
            fill='toself',
            fillcolor=cluster_avg['color'],
            line=dict(color=cluster_avg['color'], width=2),
            opacity=0.4,
            name=f'Cluster {player_cluster} Average'
        ))
        
        # Add player (bold, contrasting color)
        fig_radar_interactive.add_trace(go.Scatterpolar(
            r=player_values + player_values[:1],
            theta=feature_labels + feature_labels[:1],
            fill='toself',
            fillcolor='rgba(255, 0, 127, 0.3)',  # Hot pink - contrasts with all cluster colors
            line=dict(color='#FF007F', width=3),
            name=selected_player
        ))
        
        title_text = f'<b>{selected_player}</b><br><sub>vs Cluster {player_cluster} Average</sub>'
    else:
        # Show all cluster averages
        for cluster_num, cluster_data in cluster_averages.items():
            fig_radar_interactive.add_trace(go.Scatterpolar(
                r=cluster_data['values'] + cluster_data['values'][:1],
                theta=feature_labels + feature_labels[:1],
                fill='toself',
                fillcolor=cluster_data['color'],
                line=dict(color=cluster_data['color'], width=2),
                opacity=0.5,
                name=f'Cluster {cluster_num} ({cluster_data["size"]} players)'
            ))
        
        title_text = '<b>All Cluster Averages</b><br><sub>Select a player to compare</sub>'
    
    fig_radar_interactive.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['20', '40', '60', '80', '100'],
                showticklabels=True
            )
        ),
        title=dict(text=title_text, font=dict(size=14)),
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            font=dict(size=10)
        ),
        height=700,
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    st.plotly_chart(fig_radar_interactive, use_container_width=True)

st.markdown("---")

# ---------- DETAILED CLUSTER RADAR CHARTS (Grid below) ----------
st.markdown("#### 📊 Detailed Cluster Style Profiles")
st.markdown("Average characteristics of each playing style:")

# Calculate cluster averages for grid (reusing normalized data from above)
cluster_averages = []

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
            'color': get_cluster_color(cluster_num)
        })

# Display radar charts in grid
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
                fig_radar = go.Figure()
                
                # Add radar trace
                fig_radar.add_trace(go.Scatterpolar(
                    r=cluster_data['values'] + cluster_data['values'][:1],  # Close the polygon
                    theta=feature_labels + feature_labels[:1],
                    fill='toself',
                    fillcolor=cluster_data['color'],  # Use hex color with opacity in layout
                    line_color=cluster_data['color'],
                    opacity=0.3,  # Set fill opacity
                    name=f'Cluster {cluster_data["cluster"]}'
                ))
                
                # Update layout
                fig_radar.update_layout(
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
                
                st.plotly_chart(fig_radar, use_container_width=True)

with st.expander("📊 How to Read These Charts"):
    st.markdown("""
    **Feature Explanations:**
    - **Chance Quality**: Average xG per shot (higher = better chances)
    - **Goal Proximity**: Average X coordinate (higher = closer to goal)
    - **Movement Range**: Y position variance (higher = wider movement)
    - **Headers**: Percentage of headed shots
    - **Open Play**: Percentage from open play (less set-piece dependent)
    - **Talisman**: Share of team's shots (attacking importance)
    - **Clinical**: xG overperformance (finishing ability)
    - **Set Piece Taker**: Direct free kick percentage
    """)

st.markdown("---")

# =============================================================================
# THE STORY - DETAILED METHODOLOGY
# =============================================================================

st.header("📖 The Story: How We Got Here")

with st.expander("📊 View Dataset Overview", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shots", f"{len(df_raw):,}")
    with col2:
        st.metric("Unique Players", df_raw['player'].nunique())
    with col3:
        st.metric("Years Covered", f"{df_raw['year'].min()}-{df_raw['year'].max()}")
    with col4:
        st.metric("Players Analyzed", len(df_player))
    
    st.markdown("**Sample Data:**")
    st.dataframe(df_raw.head(10), use_container_width=True)

with st.expander("🔧 Data Processing Pipeline", expanded=False):
    st.markdown(f"""
    ### Feature Engineering Process
    
    **1. Shot Share Calculation**
    - Normalizes player impact relative to team opportunities
    - Formula: Player shots / Team shots in matches played
    
    **2. Player Aggregation**
    - Position metrics: Average X, Y standard deviation
    - Shot types: Headers, open play, set pieces
    - Performance: xG averages, goal conversion rates
    
    **3. Quality Filters**
    - Minimum {MIN_SHOTS_THRESHOLD} shots threshold
    - Removes noise from limited sample sizes
    """)

with st.expander("🎯 Feature Selection & Correlations", expanded=False):
    st.markdown("#### Feature Correlation Matrix")
    
    correlation_matrix = cluster_features_indexed.corr()
    
    fig_corr, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=False,
                linewidths=0.5,
                ax=ax)
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig_corr)
    plt.close()
    
    st.markdown("""
    **Key Insights:**
    - Low correlations indicate independent style dimensions
    - xG_avg correlates with X_avg (closer to goal = better chances)
    - Head_percent correlates with Corner_percent (expected relationship)
    """)

with st.expander("🎪 Clustering Methodology", expanded=False):
    st.markdown(f"""
    ### Why Cosine K-Means?
    
    **The Problem:** Traditional distance metrics favor high-volume players
    
    **The Solution:** Cosine similarity measures direction, not magnitude
    - Focuses on style proportions rather than absolute counts
    - Treats a player with 50 shots the same as one with 500 (if style is similar)
    - Mathematically: Compares angle between feature vectors
    
    ### Our Configuration
    - **K = {CHOSEN_K}** clusters (adjustable in sidebar)
    - **Normalization:** StandardScaler + L2 norm
    - **Initialization:** k-means++ (50 trials)
    - **Convergence:** 500 max iterations
    """)
    
    st.markdown("#### Elbow Method Analysis")
    
    # Cached elbow calculation
    @st.cache_data(show_spinner=False)
    def calculate_elbow(X, k_range):
        inertias = []
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X)
            inertias.append(kmeans_temp.inertia_)
        return inertias
    
    k_range = range(2, 13)
    inertias = calculate_elbow(X_cosine, k_range)
    
    fig_elbow, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
    ax.axvline(x=CHOSEN_K, color='r', linestyle='--', label=f'Chosen K={CHOSEN_K}')
    ax.set_title('Elbow Method for Optimal K')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig_elbow)
    plt.close()

with st.expander("📈 Alternative Visualizations", expanded=False):
    st.markdown("#### PCA Projection")
    
    fig_pca, ax = plt.subplots(figsize=(12, 8))
    
    for i in range(CHOSEN_K):
        mask = (df_player['cluster'] == i)
        ax.scatter(df_player[mask]['pca_1'], 
                  df_player[mask]['pca_2'],
                  label=f'Cluster {i}',
                  color=get_cluster_color(i),  # Use consistent color mapping
                  alpha=0.6,
                  s=30)
    
    ax.set_title(f'Player Clusters - PCA Visualization')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(title='Cluster')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_pca)
    plt.close()
    
    st.markdown("#### PCA Component Analysis")
    feature_names = cluster_features_indexed.columns
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top contributors to PC1:**")
        st.dataframe(pca_components['PC1'].abs().sort_values(ascending=False).head(5))
    with col2:
        st.markdown("**Top contributors to PC2:**")
        st.dataframe(pca_components['PC2'].abs().sort_values(ascending=False).head(5))

with st.expander("🔍 Cluster Analysis & Top Players", expanded=False):
    for cluster_num in sorted(df_player['cluster'].unique()):
        st.markdown(f"### Cluster {cluster_num}")
        cluster_df = df_player[df_player['cluster'] == cluster_num]
        top_players = cluster_df.sort_values('total_shots', ascending=False).head(10)
        
        st.dataframe(
            top_players[['player', 'total_shots', 'xG_avg', 'shot_share', 
                        'Openplay_percent', 'Head_percent']],
            use_container_width=True
        )

# =============================================================================
# INTERACTIVE TOOLS
# =============================================================================

st.header("🛠️ Interactive Tools")

# Player comparison
st.markdown("### Compare Player Styles")

col1, col2 = st.columns(2)

with col1:
    player1 = st.selectbox(
        "Select Player 1", 
        df_player['player'].sort_values().tolist(),
        index=df_player['player'].tolist().index('Lionel Messi') if 'Lionel Messi' in df_player['player'].tolist() else 0
    )

with col2:
    player2 = st.selectbox(
        "Select Player 2", 
        df_player['player'].sort_values().tolist(),
        index=df_player['player'].tolist().index('Cristiano Ronaldo') if 'Cristiano Ronaldo' in df_player['player'].tolist() else 1
    )

if player1 and player2:
    player1_idx = df_player[df_player['player'] == player1].index[0]
    player2_idx = df_player[df_player['player'] == player2].index[0]
    
    cluster1 = df_player.loc[player1_idx, 'cluster']
    cluster2 = df_player.loc[player2_idx, 'cluster']
    
    values1 = [radar_normalized.loc[player1_idx, feature] for feature in radar_features]
    values2 = [radar_normalized.loc[player2_idx, feature] for feature in radar_features]
    
    fig_compare = go.Figure()
    
    fig_compare.add_trace(go.Scatterpolar(
        r=values1 + values1[:1],
        theta=feature_labels + feature_labels[:1],
        fill='toself',
        name=f"{player1} (Cluster {cluster1})",
        line_color=cluster_colors[cluster1]
    ))
    
    fig_compare.add_trace(go.Scatterpolar(
        r=values2 + values2[:1],
        theta=feature_labels + feature_labels[:1],
        fill='toself',
        name=f"{player2} (Cluster {cluster2})",
        line_color=cluster_colors[cluster2]
    ))
    
    fig_compare.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"Style Comparison: {player1} vs {player2}",
        height=500
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)

# Player similarity finder
st.markdown("### 🔎 Find Similar Players")

target_player = st.selectbox(
    "Select Target Player", 
    df_player['player'].sort_values().tolist(),
    key='similarity_finder'
)

if target_player:
    target_cluster = df_player[df_player['player'] == target_player]['cluster'].values[0]
    
    similar_players = df_player[
        (df_player['cluster'] == target_cluster) & 
        (df_player['player'] != target_player)
    ].copy()
    
    target_features = cluster_features_indexed.loc[target_player].values
    
    def calculate_distance(player_row):
        player_features = cluster_features_indexed.loc[player_row['player']].values
        return np.linalg.norm(target_features - player_features)
    
    similar_players['distance'] = similar_players.apply(calculate_distance, axis=1)
    similar_players = similar_players.sort_values('distance').head(10)
    
    st.markdown(f"**Players with similar style to {target_player} (Cluster {target_cluster}):**")
    
    cols = st.columns(5)
    for idx, (_, row) in enumerate(similar_players.iterrows()):
        if idx < 10:
            with cols[idx % 5]:
                st.metric(
                    row['player'],
                    f"{row['total_shots']} shots",
                    delta=f"Similarity: {1/(1+row['distance']):.2f}"
                )

# =============================================================================
# DOWNLOAD & FOOTER
# =============================================================================

st.markdown("---")

st.header("💾 Export Results")

csv = df_player.to_csv(index=False)
st.download_button(
    label="📥 Download Player Clusters (CSV)",
    data=csv,
    file_name="player_clusters.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("""
**Data Science Portfolio Project** | Football Player Style Clustering  
*Built with Python, Scikit-learn, and Streamlit*
""")

# =============================================================================
# APPENDIX: ALTERNATIVE VISUALIZATIONS (PASTE AT END OF SCRIPT)
# =============================================================================

st.markdown("---")
st.header("🧪 Experimental: Clearer Visualizations")
st.markdown("Since 8 clusters can be messy in a single radar chart, here are two cleaner ways to compare them.")

col_alt_1, col_alt_2 = st.columns([1, 1])


# --- OPTION 2: THE SELECTIVE RADAR (Best for direct comparison) ---
with col_alt_2:
    st.subheader("2. Cluster Comparator")
    st.markdown("Select specific clusters to overlay (instead of all 8).")
    
    # 1. Multiselect Control
    selected_clusters = st.multiselect(
        "Select Clusters to Compare:",
        options=range(CHOSEN_K),
        default=[0, 1], # Start with just 2 for clarity
        key='alt_radar_select'
    )
    
    # 2. Plot Custom Radar
    fig_comp = go.Figure()
    
    for c_id in selected_clusters:
        data = cluster_averages[c_id]
        color_hex = data['color']
        
        # Convert hex to rgba for fill opacity
        # (Assuming color_hex is like '#RRGGBB')
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.3)"
        
        fig_comp.add_trace(go.Scatterpolar(
            r=data['values'] + data['values'][:1],
            theta=feature_labels + feature_labels[:1],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color=color_hex, width=3),
            name=f"Cluster {c_id}"
        ))
        
    fig_comp.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        height=500,
        margin=dict(l=40, r=40, t=30, b=40)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.header("🏁 Global Cluster Comparison (8-Feature Z-Scores)")

# 1. THE DEFINITIVE LIST (Must match your df_player column names exactly)
z_features = [
    'xG_avg', 
    'X_avg', 
    'Y_std', 
    'Head_percent', 
    'Openplay_percent', 
    'shot_share', 
    'avgxGoverperformance', 
    'DirectFreekick_percent'
]

# Mapping names for the UI to make them look professional
z_labels = {
    'xG_avg': 'Chance Quality',
    'X_avg': 'Goal Proximity',
    'Y_std': 'Movement Range',
    'Head_percent': 'Heading Ability',
    'Openplay_percent': 'Open Play Style',
    'shot_share': 'Talisman Index',
    'avgxGoverperformance': 'Clinical Finishing',
    'DirectFreekick_percent': 'Set Piece Threat'
}

# 2. Safety Check: Only use features that actually exist in the dataframe
available_features = [f for f in z_features if f in df_player.columns]

# 3. Calculate Stats
pop_mean = df_player[available_features].mean()
pop_std = df_player[available_features].std()

all_cluster_data = []

for cluster_id in sorted(df_player['cluster'].unique()):
    # Get famous player
    famous_player = df_player[df_player['cluster'] == cluster_id].sort_values('total_shots', ascending=False).iloc[0]['player']
    
    # Calculate Cluster Z-Scores
    cluster_avg = df_player[df_player['cluster'] == cluster_id][available_features].mean()
    z_scores = (cluster_avg - pop_mean) / pop_std
    
    for feat in available_features:
        all_cluster_data.append({
            'Cluster': f"Cluster {cluster_id}: {famous_player}s",
            'Metric': z_labels[feat],
            'Z-Score': z_scores[feat],
            'Color': 'Positive' if z_scores[feat] > 0 else 'Negative'
        })

facet_df = pd.DataFrame(all_cluster_data)

# 4. Create Plot
fig_facet = px.bar(
    facet_df,
    x='Z-Score',
    y='Metric',
    facet_col='Cluster',
    facet_col_wrap=3,
    orientation='h',
    color='Color',
    color_discrete_map={'Positive': '#00e676', 'Negative': '#ff5252'},
    text_auto='.1f',
    # This line ensures the order is identical in every chart
    category_orders={"Metric": list(z_labels.values())} 
)

fig_facet.update_layout(
    height=900, # Tall enough to see all 8 rows clearly
    xaxis=dict(range=[-3, 3]), # Standardize the scale
    showlegend=False
)

# Remove "Cluster=" prefix from titles
fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

st.plotly_chart(fig_facet, use_container_width=True)
