import pandas as pd
import plotly.express as px
import streamlit as st
import utils # Make sure utils is imported

# --- State Management and Sidebar ---
# This block MUST be at the top of every page
st.markdown(utils.load_css(), unsafe_allow_html=True)
selected_sheet = utils.add_sidebar()

# Check if the selected sheet has changed or if data is not loaded
if 'data_loaded' not in st.session_state or st.session_state.get('loaded_sheet') != selected_sheet:
    # If the sheet is changed on a different page, this will trigger the reload
    utils.initialize_app(selected_sheet)

# Final check for data loading errors before stopping the script
if 'df_enhanced' not in st.session_state or st.session_state.df_enhanced is None:
    st.warning("Data could not be loaded. Please check the sheet name and data format.")
    st.stop()

# Get the dataframe and models from session state
df = st.session_state['df_enhanced']
models = st.session_state['models']
st.header("Advanced Analytics")

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
    
    # Filter for numeric columns that actually exist in the dataframe
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