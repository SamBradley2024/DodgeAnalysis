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
# This block MUST be at the top of every page script
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Initialize state keys if they don't exist
if 'loaded_sheet' not in st.session_state:
    st.session_state.loaded_sheet = None

utils.add_sidebar() # Draws the sidebar and populates st.session_state.selected_sheet

# The definitive check: reload if the selected sheet is different from the loaded one
if st.session_state.selected_sheet != st.session_state.loaded_sheet:
    utils.initialize_app(st.session_state.selected_sheet)

# --- App Homepage ---
st.markdown("""
<div class="main-header">
    <h1>🤾 Advanced Dodgeball Analytics Dashboard</h1>
    <p>Professional-grade performance analysis with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

# Use the 'loaded_sheet' to show what data is currently displayed
st.header(f"Displaying Analysis for: `{st.session_state.get('loaded_sheet', 'N/A')}`")

if st.session_state.get('df_enhanced') is None:
    st.error("Data could not be loaded. Please select a valid worksheet from the sidebar and ensure it contains data.")
    st.stop()
else:
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