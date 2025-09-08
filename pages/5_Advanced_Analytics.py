import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- State Management and Sidebar ---
st.set_page_config(page_title="Advanced Analytics", page_icon="📊", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# Get the dataframe and models from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("Advanced Analytics")
st.info(f"Analyzing data from: **{st.session_state.source_name}**")

tab1, tab2 = st.tabs(["Player Specialization", "Statistical Correlation"])

with tab1:
    # UPDATED: Added a subheader with a help icon for explanation
    st.subheader(
        "Player Specialization Analysis",
        help="""
        This analysis identifies players with standout skills compared to the league average.

        - **Heatmap:** Shows a "Specialization Score". A score of **1.0** is exactly league average. A score of **2.0** means the player is twice as good as the average in that specific skill.
        
        - **Top Specialists:** The tables highlight the top 5 players for key aggregated skills like overall defense, offense, and game impact.
        """
    )
    utils.create_specialization_analysis(df)

with tab2:
    st.subheader("Statistical Correlations")
    st.write("Understand which individual statistics have the strongest relationship with the `Overall_Performance` score.")

    numeric_cols = [
        'Hits', 'Throws', 'Dodges', 'Catches', 'Blocks',
        'Hit_Accuracy', 'Defensive_Efficiency', 'Offensive_Rating', 'Defensive_Rating',
        'Overall_Performance'
    ]
    
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if 'Overall_Performance' in df.columns:
        performance_corr = df[existing_numeric_cols].corr()['Overall_Performance'].abs().sort_values(ascending=False).drop('Overall_Performance').head(5)
        
        if not performance_corr.empty:
            corr_df = pd.DataFrame({
                'Metric': performance_corr.index,
                'Correlation': performance_corr.values
            })
            fig = px.bar(corr_df, x='Correlation', y='Metric',
                         title='Top 5 Metrics Correlated with Overall Performance',
                         color='Correlation', color_continuous_scale='plasma',
                         text_auto='.2f')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 A higher correlation value means that a change in that metric is more likely to result in a change in the player's overall performance score.")
        else:
            st.warning("Could not calculate correlations. The dataset might be too small or lack variation.")
    else:
        st.warning("The 'Overall_Performance' column is not available, so correlations cannot be calculated.")