import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Match Analysis", page_icon="🎲", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# If data is loaded, get it from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("🎲 Match Analysis")
st.info(f"Analyzing matches from: **{st.session_state.source_name}**")

match_list = sorted(df['Match_ID'].unique())
if not match_list:
    st.warning("No matches found in the selected data source.")
    st.stop()

selected_match = st.selectbox("Select a Match to Analyze", match_list)

if selected_match:
    match_df = df[df['Match_ID'] == selected_match].copy()
    
    # --- Match Winner and Score ---
    if 'Match_Winner' in match_df.columns:
        match_winner = match_df['Match_Winner'].iloc[0]
        game_scores = match_df[match_df['Game_Outcome'] == 'Win']['Team'].value_counts()
        
        st.subheader(f"Match Result: {match_winner} Wins")
        
        cols = st.columns(len(game_scores))
        for i, (team, score) in enumerate(game_scores.items()):
            with cols[i]:
                st.metric(f"{team} Game Wins", score)
    else:
        st.subheader("Match Result")
        st.warning("Match winner could not be determined. Ensure the match has two teams.")

    st.markdown("---")

    # --- View Toggle ---
    view_mode = st.radio(
        "Select Analysis View",
        ("Match Summary", "Individual Player Performance"),
        horizontal=True,
        label_visibility="collapsed"
    )

    if view_mode == "Match Summary":
        st.subheader("Game-by-Game Breakdown")
        games_in_match = sorted(match_df['Game_ID'].unique())
        
        for game_id in games_in_match:
            game_df = match_df[match_df['Game_ID'] == game_id]
            
            # Find winner, handle cases with no winner recorded
            winner_series = game_df[game_df['Game_Outcome'] == 'Win']['Team']
            winner = winner_series.iloc[0] if not winner_series.empty else "Draw/Unknown"
            game_num = game_df['Game_Num_In_Match'].iloc[0]
            
            with st.expander(f"**Game {game_num}: {winner} won**"):
                mvp = game_df.loc[game_df['Overall_Performance'].idxmax()]
                st.write(f"**Game MVP:** {mvp['Player_ID']} (Performance: {mvp['Overall_Performance']:.2f})")
                
                player_stats = game_df[['Player_ID', 'Team', 'Overall_Performance', 'Hits', 'Catches', 'Times_Eliminated']]
                st.dataframe(player_stats.sort_values("Overall_Performance", ascending=False), use_container_width=True)

    elif view_mode == "Individual Player Performance":
        st.subheader("Match Performance Leaderboard")
        
        # Calculate average stats for each player across all games in THIS match
        match_player_summary = match_df.groupby(['Player_ID', 'Team']).agg(
            Avg_Match_Performance=('Overall_Performance', 'mean'),
            Total_Hits=('Hits', 'sum'),
            Total_Catches=('Catches', 'sum'),
            Total_Dodges=('Dodges', 'sum'),
            Total_Blocks=('Blocks', 'sum'),
            Total_Eliminations=('Times_Eliminated', 'sum'),
            Avg_KD_Ratio=('K/D_Ratio', 'mean')
        ).sort_values('Avg_Match_Performance', ascending=False).reset_index()

        st.dataframe(match_player_summary.style.format({
            'Avg_Match_Performance': '{:.2f}',
            'Avg_KD_Ratio': '{:.2f}'
        }), use_container_width=True)

        # Add a bar chart for visualization
        fig = px.bar(
            match_player_summary,
            x='Player_ID',
            y='Avg_Match_Performance',
            color='Team',
            title='Player Performance Scores for the Match',
            labels={'Player_ID': 'Player', 'Avg_Match_Performance': 'Average Performance in Match'}
        )
        st.plotly_chart(fig, use_container_width=True)