import streamlit as st
import utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Dodgeball Analytics",
    page_icon="🤾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (applied across all pages)
st.markdown(utils.load_css(), unsafe_allow_html=True)


def initialize_app():
    """
    Loads data and trains models, storing them in the session state.
    This runs only once at the start of the session.
    """
    with st.spinner("Analyzing data and training AI models... This may take a moment."):
        df = utils.load_and_enhance_data('dodgeball_data2.csv')
        if df is not None:
            # Use a copy to avoid caching issues with mutation
            df_enhanced, models = utils.train_advanced_models(df.copy())
            st.session_state['df_enhanced'] = df_enhanced
            st.session_state['models'] = models
            st.session_state['data_loaded'] = True


# --- Main Application Logic ---
# Check if data is already loaded. If not, load it.
if 'data_loaded' not in st.session_state:
    initialize_app()

# --- App Homepage ---
st.markdown("""
<div class="main-header">
    <h1>🤾 Advanced Dodgeball Analytics Dashboard</h1>
    <p>Professional-grade performance analysis with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

st.header("Welcome to the Dashboard!")
st.info("Select an analysis page from the sidebar on the left to begin exploring dodgeball data.")
st.markdown("---")
st.subheader("Dashboard Features:")
st.markdown("""
- **🏠 League Overview**: High-level statistics, leaderboards, and team comparisons.
- **👤 Player Analysis**: Deep dive into individual player performance, trends, and skills.
- **🏆 Team Analysis**: Analyze team composition, performance, and player roles.
- **🤖 AI Insights**: Get automated, data-driven insights from the entire league dataset.
- **📊 Advanced Analytics**: Explore player specialization and statistical correlations.
- **🧑‍🏫 Coaching Corner**: Receive AI-powered coaching advice for players and teams.
""")

# Check for data loading errors after initialization
if 'data_loaded' not in st.session_state or st.session_state['df_enhanced'] is None:
    st.error("There was an error loading the data. Please ensure `dodgeball_data2.csv` is in the correct directory.")
    st.stop()