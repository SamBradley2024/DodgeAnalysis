import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import utils 

# --- State Management and Sidebar ---
st.markdown(utils.load_css(), unsafe_allow_html=True)
utils.add_sidebar() 

# Check if data needs to be loaded or reloaded
if st.session_state.get('data_needs_reload', False) or 'data_loaded' not in st.session_state:
    utils.initialize_app(st.session_state.selected_sheet)

# Final check for data loading before page content
if not st.session_state.get('data_loaded', False):
    st.warning("Please select a valid worksheet from the sidebar to load the data.")
    st.stop()

# Get the dataframe and models from session state
df = st.session_state.df_enhanced
models = st.session_state.models
st.header("🎲 Single Game Analysis")
st.write("Select a specific game to see a detailed breakdown of team and player performance.")

game_list = sorted(df['Game_ID'].unique())
selected_game = st.selectbox("Select a Game to Analyze", game_list)

if selected_game:
    game_df = df[df['Game_ID'] == selected_game].copy()
    teams_in_game = game_df['Team'].unique()
    if len(teams_in_game) < 2:
        st.warning("This game does not have data for two opposing teams. Unable to compare.")
        st.stop()
    
    team1_name, team2_name = teams_in_game[0], teams_in_game[1]
    st.markdown(f"### Analysis for {selected_game}: **{team1_name}** vs **{team2_name}**")

    view_mode = st.radio("Choose Analysis View", ["Team Comparison", "Individual Player Stats"], horizontal=True)

    if view_mode == "Team Comparison":
        st.subheader("Team Head-to-Head")
        
        # UPDATED: Added more stats to aggregation
        team_stats = game_df.groupby('Team').agg(
            Total_Hits=('Hits', 'sum'),
            Total_Throws=('Throws', 'sum'),
            Total_Catches=('Catches', 'sum'),
            Total_Dodges=('Dodges', 'sum'),
            Total_Blocks=('Blocks', 'sum'),
            Total_Eliminations=('Times_Eliminated', 'sum'),
            Avg_Performance=('Overall_Performance', 'mean')
        ).reindex([team1_name, team2_name])

        col1, col2 = st.columns(2)

        for i, team_name in enumerate([team1_name, team2_name]):
            with (col1 if i == 0 else col2):
                team_outcome = game_df[game_df['Team'] == team_name]['Game_Outcome'].iloc[0]
                outcome_emoji = "🏆" if team_outcome == "Win" else "💔"
                st.markdown(f"<h5>{team_name} ({team_outcome}) {outcome_emoji}</h5>", unsafe_allow_html=True)
                st.metric("Total Hits", f"{team_stats.loc[team_name]['Total_Hits']:.0f}")
                st.metric("Total Catches", f"{team_stats.loc[team_name]['Total_Catches']:.0f}")
                st.metric("Total Throws", f"{team_stats.loc[team_name]['Total_Throws']:.0f}")
                st.metric("Total Dodges", f"{team_stats.loc[team_name]['Total_Dodges']:.0f}")
                st.metric("Total Blocks", f"{team_stats.loc[team_name]['Total_Blocks']:.0f}")
                st.metric("Total Eliminations", f"{team_stats.loc[team_name]['Total_Eliminations']:.0f}")
        
        st.markdown("---")
        
        stats_to_compare = ['Total_Hits', 'Total_Catches', 'Total_Throws', 'Total_Dodges', 'Total_Blocks']
        fig = go.Figure(data=[
            go.Bar(name=team1_name, x=stats_to_compare, y=team_stats.loc[team1_name][stats_to_compare], marker_color='#FF6B6B'),
            go.Bar(name=team2_name, x=stats_to_compare, y=team_stats.loc[team2_name][stats_to_compare], marker_color='#4ECDC4')
        ])
        fig.update_layout(barmode='group', title_text='Team Stat Comparison')
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Individual Player Stats":
        st.subheader("Player Performance Leaderboard (MVP Ranking)")
        
        # All raw stats are now included
        player_stats_df = game_df[[
            'Player_ID', 'Team', 'Overall_Performance', 'Hits', 'Throws', 'Catches', 
            'Dodges', 'Blocks', 'Hit_Out', 'Caught_Out', 'K/D_Ratio', 'Net_Impact'
        ]].rename(columns={'Overall_Performance': 'Game Performance'})
        
        player_stats_df = player_stats_df.sort_values('Game Performance', ascending=False).reset_index(drop=True)
        st.dataframe(player_stats_df.style.format({
            'Game Performance': '{:.2f}',
            'K/D_Ratio': '{:.2f}',
            'Net_Impact': '{:.1f}'
        }), use_container_width=True)

        fig = go.Figure()
        for team in teams_in_game:
            team_df = player_stats_df[player_stats_df['Team'] == team]
            fig.add_trace(go.Bar(x=team_df['Player_ID'], y=team_df['Game Performance'], name=team))
        
        fig.update_layout(title='Player Performance Scores for the Game', xaxis_title="Player", yaxis_title="Performance Score", xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)