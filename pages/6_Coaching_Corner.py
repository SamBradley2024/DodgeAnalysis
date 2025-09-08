import streamlit as st
import utils # Make sure utils is imported

# --- State Management and Sidebar ---
# This block MUST be at the top of every page
st.markdown(utils.load_css(), unsafe_allow_html=True)
selected_sheet = utils.add_sidebar()

# Check if the selected sheet has changed or if data is not loaded
if 'data_loaded' not in st.session_state or st.session_state.get('loaded_sheet') != selected_sheet:
    # If the sheet is changed on a different page, this will trigger the reload
    utils.initialize_app(selected_sheet)

# Final check for data loading errors before stopping the script
if 'df_enhanced' not in st.session_state or st.session_state.df_enhanced is None:
    st.warning("Data could not be loaded. Please check the sheet name and data format.")
    st.stop()

# Get the dataframe and models from session state
df = st.session_state['df_enhanced']
models = st.session_state['models']
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