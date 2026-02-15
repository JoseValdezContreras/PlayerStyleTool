# -*- coding: utf-8 -*-
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

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Player Style Clustering Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. SIDEBAR CONTROLS
st.sidebar.header("Analysis Controls")
MIN_SHOTS_THRESHOLD = st.sidebar.slider("Minimum Shots Threshold", min_value=20, max_value=100, value=40)
CHOSEN_K = st.sidebar.selectbox("Number of Clusters (K)", options=[4, 5, 6, 7, 8], index=1)

# 3. SILENT DATA PROCESSING (YOUR CORE LOGIC)
@st.cache_data
def process_full_analysis(threshold, k_clusters):
    try:
        # Load data
        df = pd.read_parquet('datacompleta.parquet')
        
        # Calculate Shot Share (Your Step 1)
        df_filtered = df[['X','Y','xG','player','situation','shotType','GOAL','match_id','team ']]
        df_with_team = df_filtered.copy()
        df_with_team['match_team_id'] = df_with_team['match_id'].astype(str) + '_' + df_with_team['team '].astype(str)
        shots_per_match_team = df_with_team.groupby('match_team_id').size().reset_index(name='team_shots_in_match')
        player_match_unique = df_with_team[['player', 'match_team_id']].drop_duplicates()
        player_match_unique = player_match_unique.merge(shots_per_match_team, on='match_team_id', how='left')
        player_normalized_shots = player_match_unique.groupby('player').agg({'team_shots_in_match': 'sum'}).reset_index()
        player_shot_counts = df_with_team.groupby('player').size().reset_index(name='player_shots')
        player_normalized_shots = player_normalized_shots.merge(player_shot_counts, on='player', how='left')
        player_normalized_shots['shot_share'] = player_normalized_shots['player_shots'] / player_normalized_shots['team_shots_in_match']

        # Aggregate Player Features (Your Step 2)
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
        ).reset_index()
        
        df_player = df_player.merge(player_normalized_shots[['player', 'shot_share']], on='player', how='left')
        df_player = df_player[df_player['total_shots'] >= threshold]
        df_player['avgxGoverperformance'] = (df_player.Goal_sum - df_player.xG_sum) / df_player.total_shots

        # Prepare for Clustering
        cluster_cols = ['X_avg', 'Y_std', 'xG_avg', 'Head_percent', 'Openplay_percent', 'DirectFreekick_percent', 'shot_share', 'avgxGoverperformance']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_player[cluster_cols])
        X_cosine = Normalizer(norm='l2').fit_transform(X_scaled)
        
        # Apply Clustering
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_player['cluster'] = kmeans.fit_predict(X_cosine)
        
        return df_player, X_cosine, cluster_cols
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

df_player, X_cosine, cluster_cols = process_full_analysis(MIN_SHOTS_THRESHOLD, CHOSEN_K)

# 4. RESULTS SECTION: SHOWING THE "ANSWERS" FIRST
st.title("⚽ Player Style Clustering Analysis")
st.markdown("### Executive Summary: The Results")

if df_player is not None:
    # A. t-SNE Plot (Synchronized Colors)
    st.header("📈 The Player Style Map (t-SNE)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_cosine)
    
    fig_tsne, ax_tsne = plt.subplots(figsize=(12, 7))
    scatter = ax_tsne.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df_player['cluster'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ax=ax_tsne, label='Cluster ID')
    ax_tsne.set_title("How the Machine Groups Players by Style")
    st.pyplot(fig_tsne)

    # B. Radar Charts (Synchronized Colors + Famous Names)
    st.header("📊 Average Style Profiles by Cluster")
    radar_features = ['xG_avg', 'X_avg', 'Y_std', 'Head_percent', 'Openplay_percent', 'shot_share', 'avgxGoverperformance', 'DirectFreekick_percent']
    radar_labels = ['Chance Quality', 'Goal Proximity', 'Movement Range', 'Headers', 'Open Play', 'Talisman Share', 'Clinical Finishing', 'Set Piece Taker']
    
    # Normalize values for the radar chart (0 to 1 scale)
    radar_norm = pd.DataFrame()
    for f in radar_features:
        radar_norm[f] = (df_player[f] - df_player[f].min()) / (df_player[f].max() - df_player[f].min())

    viridis_cmap = plt.cm.get_cmap('viridis', CHOSEN_K)
    
    cols = st.columns(3)
    for i in range(CHOSEN_K):
        cluster_mask = df_player['cluster'] == i
        # Find "Famous" Player (one with the most shots in that cluster)
        famous_player = df_player[cluster_mask].sort_values('total_shots', ascending=False).iloc[0]['player']
        avg_vals = radar_norm[cluster_mask][radar_features].mean().tolist()
        
        with cols[i % 3]:
            rgb = viridis_cmap(i)
            hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_vals + avg_vals[:1],
                theta=radar_labels + radar_labels[:1],
                fill='toself',
                fillcolor=hex_color,
                opacity=0.4,
                line_color=hex_color,
                name=f"Cluster {i}"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"Cluster {i}: (The {famous_player}s)",
                height=350, margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # 5. METHODOLOGY SECTION (TUCKED INTO EXPANDERS)
    st.divider()
    st.header("📖 How it Works: Methodology")
    
    with st.expander("Step 1 & 2: Data Cleaning & Feature Engineering"):
        st.write("Calculated Shot Share and normalized player performance to remove volume bias.")
        st.dataframe(df_player[['player', 'total_shots', 'shot_share', 'cluster']].head(10))

    with st.expander("Step 3: Determining K (The Elbow Method)"):
        st.markdown("This graph justifies why we chose K=5 clusters.")
        inertia = []
        for k in range(2, 10):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cosine)
            inertia.append(km.inertia_)
        fig_elb, ax_elb = plt.subplots(figsize=(8, 3))
        ax_elb.plot(range(2, 10), inertia, marker='o')
        st.pyplot(fig_elb)

    # 6. INTERACTIVE TOOLS
    st.header("🔎 Explore Further")
    tab1, tab2 = st.tabs(["Compare Two Players", "Find Similar Players"])
    
    with tab1:
        st.subheader("Direct Style Overlay")
        p1 = st.selectbox("Select Player 1", df_player['player'].sort_values(), index=0)
        p2 = st.selectbox("Select Player 2", df_player['player'].sort_values(), index=1)
        # You can copy your original Plotly radar comparison logic here using p1 and p2
        st.info("Select players to see how their shapes overlap.")

    with tab2:
        st.subheader("Player Finder")
        target = st.selectbox("Find players similar to:", df_player['player'].sort_values())
        # Add your Euclidean distance similarity logic here
        st.write(f"Showing results for players in the same cluster as {target}.")

    # Download Button
    st.sidebar.markdown("---")
    csv = df_player.to_csv(index=False)
    st.sidebar.download_button("Download Full Cluster Data", csv, "player_clusters.csv", "text/csv")
    # 6. INTERACTIVE TOOLS
    st.header("🔎 Explore Further")
    tab1, tab2 = st.tabs(["Compare Two Players", "Find Similar Players"])
    
    with tab1:
        st.subheader("Direct Style Overlay")
        p1 = st.selectbox("Select Player 1", df_player['player'].sort_values(), index=0)
        p2 = st.selectbox("Select Player 2", df_player['player'].sort_values(), index=1)
        # You can copy your original Plotly radar comparison logic here using p1 and p2
        st.info("Select players to see how their shapes overlap.")

    with tab2:
        st.subheader("Player Finder")
        target = st.selectbox("Find players similar to:", df_player['player'].sort_values())
        # Add your Euclidean distance similarity logic here
        st.write(f"Showing results for players in the same cluster as {target}.")

    # Download Button
    st.sidebar.markdown("---")
    csv = df_player.to_csv(index=False)
    st.sidebar.download_button("Download Full Cluster Data", csv, "player_clusters.csv", "text/csv")
