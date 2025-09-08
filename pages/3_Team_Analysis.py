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
st.header("Team Analysis")

team_list = sorted(df['Team'].unique())
selected_team = st.selectbox("Select Team", team_list)

if selected_team:
    team_data = df[df['Team'] == selected_team]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        utils.styled_metric("Team Size", team_data['Player_ID'].nunique())
    with col2:
        win_rate = (team_data['Game_Outcome'] == 'Win').mean()
        utils.styled_metric("Team Win Rate", f"{win_rate:.1%}")
    with col3:
        avg_perf = team_data['Overall_Performance'].mean()
        utils.styled_metric("Avg Performance", f"{avg_perf:.2f}")
    with col4:
        # Get the most common player role on the team
        dominant_role = team_data['Player_Role'].mode()[0]
        utils.styled_metric("Dominant Role", dominant_role)

    fig = utils.create_team_analytics(df, selected_team)
    st.plotly_chart(fig, use_container_width=True)