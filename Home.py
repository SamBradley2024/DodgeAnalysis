import streamlit as st
import utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Dodgeball Analytics",
    page_icon="🤾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(utils.load_css(), unsafe_allow_html=True)
# Add the sidebar to the page and get the selected sheet
selected_sheet = utils.add_sidebar()

def initialize_app(worksheet_name):
    """
    Loads data from a specific worksheet and trains models,
    storing them in the session state.
    """
    with st.spinner(f"Loading data from '{worksheet_name}' and training models..."):
        df = utils.load_and_enhance_data(worksheet_name)
        if df is not None:
            df_enhanced, models = utils.train_advanced_models(df.copy())
            st.session_state['df_enhanced'] = df_enhanced
            st.session_state['models'] = models
            st.session_state['data_loaded'] = True
            st.session_state['loaded_sheet'] = worksheet_name # Track which sheet is loaded

# --- Main Application Logic ---
# Check if data needs to be loaded or reloaded
if 'data_loaded' not in st.session_state or st.session_state.get('loaded_sheet') != selected_sheet:
    initialize_app(selected_sheet)

# --- App Homepage ---
st.markdown("""
<div class="main-header">
    <h1>🤾 Advanced Dodgeball Analytics Dashboard</h1>
    <p>Professional-grade performance analysis with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

st.header(f"Displaying Analysis for: `{selected_sheet}`")
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

# Check for data loading errors
if 'data_loaded' not in st.session_state or st.session_state.get('df_enhanced') is None:
    st.error("There was an error loading the data. Please ensure your secrets are configured and the sheet is shared and named correctly.")
    st.stop()