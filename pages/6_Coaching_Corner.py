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