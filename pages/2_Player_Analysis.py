import streamlit as st
import utils
import plotly.graph_objects as go

# --- State Management and Sidebar ---
# This block MUST be at the top of every page script
st.set_page_config(page_title="Player Analysis", page_icon="👤", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# If data is loaded, get it from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("Individual Player Analysis")
st.write(f"Displaying data from worksheet: **{st.session_state.loaded_sheet}**")

# --- Player Selection ---
player_list = sorted(df['Player_ID'].unique())
if not player_list:
    st.warning("No players found in the selected worksheet.")
    st.stop()
    
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    # Create a summary dataframe with one row per player to easily access aggregated stats
    player_summary = df.groupby('Player_ID').first()
    player_summary_data = player_summary.loc[selected_player]

    # --- Key Metrics ---
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

    # --- Main Visual ---
    fig = utils.create_player_dashboard(df, selected_player)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # --- Elimination Profile Section ---
    st.subheader("Elimination Profile")
    col1, col2 = st.columns([1, 2])

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

        if hit_out_total > caught_out_total * 1.5:
            st.warning("Insight: This player is more frequently eliminated by being hit directly. Coaching should focus on improving dodging ability and spatial awareness.")
        elif caught_out_total > hit_out_total * 1.5:
            st.warning("Insight: This player is more frequently eliminated by having their throws caught. Coaching should focus on shot selection, throwing power, and avoiding risky throws to well-positioned catchers.")
        else:
            st.info("Insight: This player has a balanced elimination profile, showing no specific defensive weakness.")

    st.markdown("---")
    
    # --- Player Comparison ---
    st.subheader("Player Comparison")
    if st.checkbox("Enable Player Comparison"):
        comparison_player = st.selectbox(
            "Compare with:",
            [p for p in player_list if p != selected_player],
            key="comparison_player_select"
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