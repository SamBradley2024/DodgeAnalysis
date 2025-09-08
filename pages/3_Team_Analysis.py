import streamlit as st
import utils

st.set_page_config(page_title="Team Analysis", page_icon="🏆", layout="wide")

if 'df_enhanced' not in st.session_state:
    st.warning("Please go to the Home page to load the data first.")
    st.stop()

df = st.session_state['df_enhanced']

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