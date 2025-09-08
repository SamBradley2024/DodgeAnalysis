import streamlit as st
import utils # Make sure utils is imported

# Add the sidebar to the page and get the selected sheet
selected_sheet = utils.add_sidebar()

# --- Data Loading and State Management ---
# Check if data is loaded
if 'df_enhanced' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Data not loaded. Please go to the Home page to load data.")
    st.stop()

# Check if the selected sheet in the sidebar is the one that's loaded
if st.session_state.get('loaded_sheet') != selected_sheet:
    st.warning(f"You selected the '{selected_sheet}' worksheet, but the '{st.session_state.get('loaded_sheet')}' worksheet is loaded. Please go to the Home page to load the new sheet.")
    st.stop()

# Get the dataframe from session state
df = st.session_state['df_enhanced']
models = st.session_state['models']

st.header("League Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    utils.styled_metric("Total Players", df['Player_ID'].nunique())
with col2:
    utils.styled_metric("Total Teams", df['Team'].nunique())
with col3:
    utils.styled_metric("Games Analyzed", df['Game_ID'].nunique())
with col4:
    avg_performance = df['Overall_Performance'].mean()
    utils.styled_metric("Avg Performance", f"{avg_performance:.2f}")

st.plotly_chart(utils.create_league_overview(df), use_container_width=True)

st.subheader("🏆 Leaderboards")
# UPDATED: Added new tabs for more leaderboards
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overall Performance", "K/D Ratio", "Hit Accuracy",
    "Top Throwers", "Top Dodgers", "Top Blockers", "Win Rate"
])

with tab1:
    leaderboard = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Performance']].sort_values('Avg_Performance', ascending=False).head(10)
    st.dataframe(leaderboard.style.format({'Avg_Performance': '{:.2f}'}), use_container_width=True)

with tab2:
    kd_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_KD_Ratio']].sort_values('Avg_KD_Ratio', ascending=False).head(10)
    st.dataframe(kd_board.style.format({'Avg_KD_Ratio': '{:.2f}'}), use_container_width=True)

with tab3:
    accuracy_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Hit_Accuracy']].sort_values('Avg_Hit_Accuracy', ascending=False).head(10)
    st.dataframe(accuracy_board.style.format({'Avg_Hit_Accuracy': '{:.1%}'}), use_container_width=True)

with tab4:
    thrower_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Throws']].sort_values('Avg_Throws', ascending=False).head(10)
    st.dataframe(thrower_board.style.format({'Avg_Throws': '{:.2f}'}), use_container_width=True)

with tab5:
    dodger_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Dodges']].sort_values('Avg_Dodges', ascending=False).head(10)
    st.dataframe(dodger_board.style.format({'Avg_Dodges': '{:.2f}'}), use_container_width=True)

with tab6:
    blocker_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Blocks']].sort_values('Avg_Blocks', ascending=False).head(10)
    st.dataframe(blocker_board.style.format({'Avg_Blocks': '{:.2f}'}), use_container_width=True)

with tab7:
    win_rate_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Win_Rate']].sort_values('Win_Rate', ascending=False).head(10)
    st.dataframe(win_rate_board.style.format({'Win_Rate': '{:.1%}'}), use_container_width=True)