import streamlit as st
import utils

# --- State Management and Sidebar ---
st.set_page_config(page_title="Coaching Corner", page_icon="🧑‍🏫", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)
utils.add_sidebar()

# The definitive check: reload if data is missing.
utils.main_data_loader()

# Final check for data loading before page content
if 'df_enhanced' not in st.session_state or st.session_state.df_enhanced is None:
    st.warning("Data could not be loaded. Please select a valid worksheet from the sidebar.")
    st.stop()

# Get the dataframe and models from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("🧑‍🏫 Coaching Corner")
st.write(f"Get AI-powered, actionable advice based on data from **{st.session_state.loaded_sheet}**.")

coach_mode = st.radio("Select Coaching Mode", ["Player Coaching", "Team Coaching"], horizontal=True)

if coach_mode == "Player Coaching":
    player_list = sorted(df['Player_ID'].unique())
    if not player_list:
        st.warning("No players found in the selected worksheet.")
        st.stop()
        
    selected_player = st.selectbox("Select a Player to Coach", player_list)
    if selected_player:
        report, fig = utils.generate_player_coaching_report(df, selected_player)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        for line in report:
            st.markdown(line)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)

elif coach_mode == "Team Coaching":
    team_list = sorted(df['Team'].unique())
    if not team_list:
        st.warning("No teams found in the selected worksheet.")
        st.stop()
        
    selected_team = st.selectbox("Select a Team to Coach", team_list)
    if selected_team:
        report = utils.generate_team_coaching_report(df, selected_team)
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        for line in report:
            st.markdown(line)
        st.markdown('</div>', unsafe_allow_html=True)