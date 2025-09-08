import streamlit as st
import utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Dodgeball Analytics",
    page_icon="🤾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management and Sidebar ---
# This block MUST be at the top of every page
st.markdown(utils.load_css(), unsafe_allow_html=True)
selected_sheet = utils.add_sidebar()

# Check if the selected sheet has changed or if data is not loaded
if 'data_loaded' not in st.session_state or st.session_state.get('loaded_sheet') != selected_sheet:
    utils.initialize_app(selected_sheet)

# --- App Homepage ---
st.markdown("""
<div class="main-header">
    <h1>🤾 Advanced Dodgeball Analytics Dashboard</h1>
    <p>Professional-grade performance analysis with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

st.header(f"Displaying Analysis for: `{st.session_state.loaded_sheet}`")
st.success("Data successfully loaded. Select a page from the navigation bar to begin.")
st.markdown("---")
st.subheader("Dashboard Features:")
st.markdown("""
- **🏠 League Overview**: High-level statistics, leaderboards, and team comparisons.
- **👤 Player Analysis**: Deep dive into individual player performance, trends, and skills.
- **🏆 Team Analysis**: Analyze team composition, performance, and player roles.
- **🤖 AI Insights**: Get automated, data-driven insights from the entire league dataset.
- **📊 Advanced Analytics**: Explore player specialization and statistical correlations.
- **🧑‍🏫 Coaching Corner**: Receive AI-powered coaching advice for players and teams.
- **🎲 Game Analysis**: View a match-by-match breakdown of performance.
""")

# Final check for data loading errors before stopping the script
if 'df_enhanced' not in st.session_state or st.session_state.df_enhanced is None:
    st.error("There was a critical error loading the data. Please check your secrets and sheet configuration.")
    st.stop()