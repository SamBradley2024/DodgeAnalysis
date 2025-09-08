import streamlit as st
import utils 

# --- State Management and Sidebar ---
# This block MUST be at the top of every page script
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Initialize state keys if they don't exist
if 'loaded_sheet' not in st.session_state:
    st.session_state.loaded_sheet = None

utils.add_sidebar() # Draws the sidebar and populates st.session_state.selected_sheet

# The definitive check: reload if the selected sheet is different from the loaded one
if st.session_state.selected_sheet != st.session_state.loaded_sheet:
    utils.initialize_app(st.session_state.selected_sheet)

# Final check for data loading before page content
if st.session_state.get('df_enhanced') is None:
    st.warning("Data could not be loaded. Please select a valid worksheet from the sidebar.")
    st.stop()

# Get the dataframe and models from session state
df = st.session_state.df_enhanced
models = st.session_state.models
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