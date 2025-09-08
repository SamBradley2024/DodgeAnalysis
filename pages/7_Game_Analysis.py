import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import utils

# --- State Management and Sidebar ---
st.set_page_config(page_title="Game Analysis", page_icon="🎲", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# If data is loaded, get it from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("🎲 Single Game Analysis")
st.info(f"Analyzing data from: **{st.session_state.source_name}**")

game_list = sorted(df['Game_ID'].unique())
if not game_list:
    st.warning("No games found in the selected data source.")
    st.stop()

selected_game = st.selectbox("Select a Game to Analyze", game_list)

if selected_game:
    game_df = df[df['Game_ID'] == selected_game].copy()
    teams_in_game = game_df['Team'].unique()
    
    if len(teams_in_game) < 2:
        # --- View for single-team games ---
        team_name = teams_in_game[0] if len(teams_in_game) > 0 else "Unknown Team"
        st.subheader(f"Player Stats for {team_name} in Game {selected_game}")
        
        # UPDATED: Added all relevant stats to the table
        player_stats_df = game_df[['Player_ID', 'Overall_Performance', 'Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'Times_Eliminated']].rename(columns={'Overall_Performance': 'Game Performance'})
        player_stats_df = player_stats_df.sort_values('Game Performance', ascending=False).reset_index(drop=True)
        st.dataframe(player_stats_df.style.format({'Game Performance': '{:.2f}'}), use_container_width=True)

    else:
        # --- View for two-team games ---
        team1_name, team2_name = teams_in_game[0], teams_in_game[1]
        st.markdown(f"### Analysis for {selected_game}: **{team1_name}** vs **{team2_name}**")
        
        view_mode = st.radio("Choose Analysis View", ["Team Comparison", "Individual Player Stats"], horizontal=True)

        if view_mode == "Team Comparison":
            st.subheader("Team Head-to-Head")
            team_stats = game_df.groupby('Team').agg(Total_Hits=('Hits', 'sum'), Total_Throws=('Throws', 'sum'), Total_Catches=('Catches', 'sum'), Total_Dodges=('Dodges', 'sum'), Total_Blocks=('Blocks', 'sum'), Total_Eliminations=('Times_Eliminated', 'sum')).reindex([team1_name, team2_name])
            # ... (Metrics display remains the same)

            # UPDATED: Added all stats to the comparison chart
            stats_to_compare = ['Total_Hits', 'Total_Catches', 'Total_Throws', 'Total_Dodges', 'Total_Blocks']
            fig = go.Figure(data=[go.Bar(name=team1_name, x=stats_to_compare, y=team_stats.loc[team1_name][stats_to_compare], marker_color='#FF6B6B'), go.Bar(name=team2_name, x=stats_to_compare, y=team_stats.loc[team2_name][stats_to_compare], marker_color='#4ECDC4')])
            fig.update_layout(barmode='group', title_text='Team Stat Comparison')
            st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "Individual Player Stats":
            st.subheader("Player Performance Leaderboard (MVP Ranking)")
            # UPDATED: Added all relevant stats to the table
            player_stats_df = game_df[['Player_ID', 'Team', 'Overall_Performance', 'Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'Times_Eliminated', 'K/D_Ratio', 'Net_Impact']].rename(columns={'Overall_Performance': 'Game Performance'})
            player_stats_df = player_stats_df.sort_values('Game Performance', ascending=False).reset_index(drop=True)
            st.dataframe(player_stats_df.style.format({'Game Performance': '{:.2f}', 'K/D_Ratio': '{:.2f}', 'Net_Impact': '{:.1f}'}), use_container_width=True)