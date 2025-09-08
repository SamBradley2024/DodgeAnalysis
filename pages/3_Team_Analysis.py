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