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