import pandas as pd
import plotly.express as px
import utils
import streamlit as st

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