import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- State Management and Sidebar ---
st.set_page_config(page_title="Advanced Analytics", page_icon="📊", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)
utils.add_sidebar()

# The definitive check: reload if data is missing.
utils.main_data_loader()

# Final check for data loading before page content
if 'df_enhanced' not in st.session_state or st.session_state.df_enhanced is None:
    st.warning("Data could not be loaded. Please select a valid worksheet from the sidebar.")
    st.stop()

# Get the dataframe and models from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("Advanced Analytics")
st.write(f"Displaying data from worksheet: **{st.session_state.loaded_sheet}**")

tab1, tab2 = st.tabs(["Specialization Analysis", "Correlation Analysis"])

with tab1:
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
    
    performance_corr = df[existing_numeric_cols].corr()['Overall_Performance'].abs().sort_values(ascending=False)[1:6]

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