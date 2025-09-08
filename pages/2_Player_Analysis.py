import streamlit as st
import utils 
import plotly.graph_objects as go


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
st.header("Individual Player Analysis")

player_list = sorted(df['Player_ID'].unique())
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    player_summary_data = df.groupby('Player_ID').first().loc[selected_player]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        utils.styled_metric("Player Role", player_summary_data['Player_Role'])
    with col2:
        utils.styled_metric("Win Rate", f"{player_summary_data['Win_Rate']:.1%}")
    with col3:
        utils.styled_metric("Avg K/D Ratio", f"{player_summary_data['Avg_KD_Ratio']:.2f}",
                            help_text="Ratio of opponents eliminated to times this player was eliminated.")
    with col4:
        utils.styled_metric("Avg Net Impact", f"{player_summary_data['Avg_Net_Impact']:.2f}",
                            help_text="(Hits + Catches) - Times Eliminated. A measure of total impact on player count.")

    fig = utils.create_player_dashboard(df, selected_player)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # --- NEW: Elimination Profile Section ---
    st.subheader("Elimination Profile")
    col1, col2 = st.columns([1, 2]) # Give more space to the text column

    with col1:
        hit_out_total = player_summary_data['Total_Hit_Out']
        caught_out_total = player_summary_data['Total_Caught_Out']
        
        if hit_out_total + caught_out_total == 0:
            st.info("This player has not been eliminated yet.")
        else:
            elimination_labels = ['Hit Out', 'Caught Out']
            elimination_values = [hit_out_total, caught_out_total]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=elimination_labels,
                values=elimination_values,
                hole=.3,
                marker_colors=['#FF6B6B', '#FECA57']
            )])
            fig_pie.update_layout(title_text="Breakdown of Eliminations", showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.write(f"Analyzing how **{selected_player}** gets eliminated provides insight into their defensive vulnerabilities.")
        st.metric("Total Times Eliminated by Being Hit", f"{hit_out_total:.0f}")
        st.metric("Total Times Eliminated by a Catch", f"{caught_out_total:.0f}")

        if hit_out_total > caught_out_total:
            st.warning("Insight: This player is more frequently eliminated by being hit directly. Coaching should focus on improving dodging ability and spatial awareness.")
        elif caught_out_total > hit_out_total:
            st.warning("Insight: This player is more frequently eliminated by having their throws caught. Coaching should focus on shot selection, throwing power, and avoiding risky throws to well-positioned catchers.")
        else:
            st.info("Insight: This player has a balanced elimination profile, showing no specific defensive weakness.")

    st.markdown("---")
    st.subheader("Player Comparison")
    # ... (rest of the file is unchanged) ...
    if st.checkbox("Enable Player Comparison"):
        comparison_player = st.selectbox(
            "Compare with:",
            [p for p in player_list if p != selected_player]
        )
        if comparison_player:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = utils.create_player_dashboard(df, selected_player)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True, key="compare_chart_1")
            with col2:
                fig2 = utils.create_player_dashboard(df, comparison_player)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True, key="compare_chart_2")