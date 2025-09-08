import streamlit as st
import utils 

# --- State Management and Sidebar ---
st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# If data is loaded, get it from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("🤖 AI-Powered Insights")
st.info(f"Analyzing data from: **{st.session_state.source_name}**")
st.markdown("---")


insights = utils.generate_insights(df)

if 'outcome_model' in models:
    model, le, accuracy, _ = models['outcome_model']
    st.markdown(f"""
    <div class="insight-box">
        <h4>Game Outcome Prediction Model</h4>
        <p>The AI model trained to predict whether a game is a 'Win' or 'Loss' based on player stats achieves <strong>{accuracy:.1%} accuracy</strong> on unseen test data. This indicates a strong relationship between in-game performance metrics and the final outcome.</p>
    </div>
    """, unsafe_allow_html=True)

for insight in insights:
    st.markdown(f"""
    <div class="insight-box">
        <p>{insight}</p>
    </div>
    """, unsafe_allow_html=True)