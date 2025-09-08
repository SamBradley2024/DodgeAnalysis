import streamlit as st
import utils

st.set_page_config(page_title="Coaching Corner", page_icon="🧑‍🏫", layout="wide")

if 'df_enhanced' not in st.session_state:
    st.warning("Please go to the Home page to load the data first.")
    st.stop()

df = st.session_state['df_enhanced']

st.header("🧑‍🏫 Coaching Corner")
st.write("Get AI-powered, actionable advice to improve player and team performance.")

coach_mode = st.radio("Select Coaching Mode", ["Player Coaching", "Team Coaching"], horizontal=True)

if coach_mode == "Player Coaching":
    player_list = sorted(df['Player_ID'].unique())
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
    selected_team = st.selectbox("Select a Team to Coach", team_list)
    if selected_team:
        report = utils.generate_team_coaching_report(df, selected_team)
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        for line in report:
            st.markdown(line)
        st.markdown('</div>', unsafe_allow_html=True)