import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
import gspread
from google.oauth2.service_account import Credentials

warnings.filterwarnings('ignore')


# --- Styling and UI Helpers ---

def load_css():
    """Returns the custom CSS string."""
    return """
    <style>
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .metric-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4ECDC4;
            margin: 0.5rem 0;
        }
        .insight-box {
            background: #e8f4fd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        .warning-box {
            background: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
    </style>
    """

def styled_metric(label, value, help_text=""):
    """Creates a styled metric box using custom CSS."""
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label, value, help=help_text)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Centralized Sidebar and Data Initialization ---

def add_sidebar():
    """Adds a sidebar with a worksheet selector to the app."""
    st.sidebar.title("Data Source")
    sheet_names = get_worksheet_names()
    
    selected_sheet = st.sidebar.selectbox(
        "Select a Worksheet (e.g., Season)",
        sheet_names,
        key='selected_sheet' 
    )
    return selected_sheet

def initialize_app(worksheet_name):
    """
    Loads data from a specific worksheet and trains models,
    storing them in the session state. This is the master function
    to call whenever data needs to be loaded or reloaded.
    """
    with st.spinner(f"Loading data from '{worksheet_name}' and training models..."):
        df = load_and_enhance_data(worksheet_name)
        if df is not None:
            # Clear previous model cache to ensure fresh models are trained
            if 'models' in st.session_state:
                del st.session_state['models']
                
            df_enhanced, models = train_advanced_models(df.copy())
            st.session_state['df_enhanced'] = df_enhanced
            st.session_state['models'] = models
            st.session_state['data_loaded'] = True
            st.session_state['loaded_sheet'] = worksheet_name # Track which sheet is loaded

# --- Google Sheets Connection Functions ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_worksheet_names():
    """Gets a list of all worksheet (tab) names from the Google Sheet."""
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        client = gspread.authorize(creds)
        spreadsheet = client.open("Dodgeball App Data")
        return [sheet.title for sheet in spreadsheet.worksheets()]
    except Exception as e:
        st.error(f"Could not retrieve worksheet names: {e}")
        return ["Sheet1"]

@st.cache_data(ttl=300) # Cache for 5 minutes
def load_and_enhance_data(worksheet_name):
    """Loads data from a specific Google Sheet worksheet and performs feature engineering."""
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        client = gspread.authorize(creds)
        sheet = client.open("Dodgeball App Data").worksheet(worksheet_name)
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

    except Exception as e:
        st.error(f"Error reading from worksheet '{worksheet_name}': {e}")
        return None

    # --- Feature engineering logic ---
    numeric_cols = ['Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'Hit_Out', 'Caught_Out']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Times_Eliminated'] = df['Hit_Out'] + df['Caught_Out']
    df['K/D_Ratio'] = df['Hits'] / df['Times_Eliminated'].replace(0, 1)
    df['Net_Impact'] = (df['Hits'] + df['Catches']) - df['Times_Eliminated']
    df['Hit_Accuracy'] = np.where(df['Throws'] > 0, df['Hits'] / df['Throws'], 0)
    df['Defensive_Efficiency'] = np.where((df['Catches'] + df['Dodges'] + df['Hit_Out']) > 0,
                                          (df['Catches'] + df['Dodges']) / (df['Catches'] + df['Dodges'] + df['Hit_Out']), 0)
    df['Offensive_Rating'] = (df['Hits'] * 2 + df['Throws'] * 0.5) / (df['Throws'] + 1)
    df['Defensive_Rating'] = (df['Dodges'] + df['Catches'] * 2) / 3
    df['Overall_Performance'] = (
        df['Offensive_Rating'] * 0.35 +
        df['Defensive_Rating'] * 0.35 +
        df['K/D_Ratio'] * 0.15 +
        df['Net_Impact'] * 0.05 +
        df['Hit_Accuracy'] * 0.05 +
        df['Defensive_Efficiency'] * 0.05
    )
    df['Game_Impact'] = np.where(df['Game_Outcome'] == 'Win',
                                 df['Overall_Performance'] * 1.2,
                                 df['Overall_Performance'] * 0.8)

    player_stats = df.groupby('Player_ID').agg(
        Avg_Performance=('Overall_Performance', 'mean'),
        Performance_Consistency=('Overall_Performance', 'std'),
        Avg_Hit_Accuracy=('Hit_Accuracy', 'mean'),
        Avg_KD_Ratio=('K/D_Ratio', 'mean'),
        Avg_Net_Impact=('Net_Impact', 'mean'),
        Avg_Throws=('Throws', 'mean'),
        Avg_Dodges=('Dodges', 'mean'),
        Avg_Blocks=('Blocks', 'mean'),
        Total_Hit_Out=('Hit_Out', 'sum'),
        Total_Caught_Out=('Caught_Out', 'sum'),
        Win_Rate=('Game_Outcome', lambda x: (x == 'Win').mean())
    ).round(3)

    player_stats['Consistency_Score'] = 1 / (player_stats['Performance_Consistency'] + 0.01)
    df = df.merge(player_stats, on='Player_ID', how='left')

    return df


# --- Advanced ML Models ---
@st.cache_resource
def train_advanced_models(_df):
    """
    Train multiple ML models for different predictions.
    Works on a copy of the dataframe to avoid side effects.
    """
    df = _df.copy()
    models = {}

    # Player Role Classification
    role_features = ['Hits', 'Throws', 'Dodges', 'Catches', 'Hit_Accuracy', 'Defensive_Efficiency', 'Offensive_Rating', 'Defensive_Rating', 'K/D_Ratio']
    df_role_features = df[role_features].dropna()

    if df_role_features.empty or len(df_role_features) < 4:
        st.warning("Not enough data to create player roles for the selected sheet.")
        return df, models

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_role_features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df.loc[df_role_features.index, 'Role_Cluster'] = kmeans.fit_predict(scaled_features)

    cluster_centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    league_average_stats = df_role_features.mean()
    role_names = []
    name_map = {
        'Hits': 'Striker', 'Throws': 'Thrower', 'Dodges': 'Evader',
        'Catches': 'Catcher', 'Hit_Accuracy': 'Accurate', 'Defensive_Efficiency': 'Efficient',
        'Offensive_Rating': 'Offensive', 'Defensive_Rating': 'Defensive', 'K/D_Ratio': 'Clutch'
    }

    for i in range(cluster_centers_unscaled.shape[0]):
        center_stats = pd.Series(cluster_centers_unscaled[i], index=role_features)
        specialization_scores = (center_stats - league_average_stats) / (league_average_stats + 1e-6)
        top_specializations = specialization_scores.nlargest(2)
        primary_spec_name = name_map.get(top_specializations.index[0], top_specializations.index[0])
        secondary_spec_name = name_map.get(top_specializations.index[1], top_specializations.index[1])
        base_role_name = f"{primary_spec_name}-{secondary_spec_name} Hybrid"
        final_role_name = base_role_name
        counter = 1
        while final_role_name in role_names:
            counter += 1
            final_role_name = f"{base_role_name} ({counter})"
        role_names.append(final_role_name)

    role_mapping = {i: role_names[i] for i in range(len(role_names))}
    df['Player_Role'] = df['Role_Cluster'].map(role_mapping)
    models['role_model'] = (kmeans, scaler, role_mapping, role_names)
    
    # Game Outcome Prediction
    outcome_features = ['Hits', 'Throws', 'Dodges', 'Catches', 'Overall_Performance', 'Offensive_Rating', 'Defensive_Rating', 'K/D_Ratio', 'Net_Impact']
    outcome_df = df.dropna(subset=outcome_features + ['Game_Outcome'])
    if len(outcome_df) > 1 and len(outcome_df['Game_Outcome'].unique()) > 1:
        le = LabelEncoder()
        y = le.fit_transform(outcome_df['Game_Outcome'])
        X = outcome_df[outcome_features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
        models['outcome_model'] = (rf_classifier, le, accuracy, outcome_features)

    # Performance Prediction
    perf_features = ['Hits', 'Throws', 'Dodges', 'Catches', 'Times_Eliminated']
    perf_df = df.dropna(subset=perf_features + ['Overall_Performance'])
    if not perf_df.empty:
        gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_regressor.fit(perf_df[perf_features], perf_df['Overall_Performance'])
        models['performance_model'] = (gb_regressor, perf_features)
        
    return df, models


# --- Visualization Functions ---
def create_player_dashboard(df, player_id):
    player_data = df[df['Player_ID'] == player_id]
    if player_data.empty:
        st.error(f"No data found for player {player_id}")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Radar', 'Game-by-Game Performance', 'Skill Distribution', 'Win Rate Analysis'),
        specs=[[{"type": "polar"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "pie"}]]
    )
    
    radar_stats = ['Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'K/D_Ratio']
    avg_radar_stats = player_data[radar_stats].mean()
    fig.add_trace(go.Scatterpolar(
        r=avg_radar_stats.values,
        theta=radar_stats,
        fill='toself', name='Avg Skills', line_color='#FF6B6B'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=player_data['Game_ID'], y=player_data['Overall_Performance'],
        mode='lines+markers', name='Performance Trend', line=dict(color='#4ECDC4', width=3)
    ), row=1, col=2)

    bar_stats_cols = ['Hits', 'Throws', 'Catches', 'Dodges', 'Blocks']
    avg_bar_stats = player_data[bar_stats_cols].mean()
    fig.add_trace(go.Bar(
        x=bar_stats_cols, y=avg_bar_stats.values,
        name='Average Stats', marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    ), row=2, col=1)

    outcomes = player_data['Game_Outcome'].value_counts()
    fig.add_trace(go.Pie(
        labels=outcomes.index, values=outcomes.values,
        name="Win Rate", marker_colors=['#4ECDC4', '#FF6B6B']
    ), row=2, col=2)
    fig.update_layout(height=800, showlegend=False, title_text=f"Comprehensive Dashboard: {player_id}")
    return fig


def create_team_analytics(df, team_id):
    team_data = df[df['Team'] == team_id]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Team Performance Distribution', 'Player Roles', 'Game Outcomes', 'Offensive vs. Defensive Rating'),
        specs=[[{"type": "histogram"}, {"type": "bar"}], [{"type": "pie"}, {"type": "scatter"}]]
    )
    fig.add_trace(go.Histogram(x=team_data['Overall_Performance'], nbinsx=15, name='Performance Distribution', marker_color='#4ECDC4'), row=1, col=1)
    
    if 'Player_Role' in team_data.columns:
        role_counts = team_data['Player_Role'].value_counts()
        fig.add_trace(go.Bar(x=role_counts.index, y=role_counts.values, name='Player Roles', marker_color='#FF6B6B'), row=1, col=2)
    
    outcomes = team_data['Game_Outcome'].value_counts()
    fig.add_trace(go.Pie(labels=outcomes.index, values=outcomes.values, name="Outcomes"), row=2, col=1)
    fig.add_trace(go.Scatter(x=team_data['Offensive_Rating'], y=team_data['Defensive_Rating'], mode='markers', text=team_data['Player_ID'], name='Off vs Def Rating', marker=dict(size=10, color=team_data['Overall_Performance'], colorscale='Viridis', showscale=True)), row=2, col=2)
    fig.update_layout(height=800, title_text=f"Team Analytics: {team_id}", showlegend=False)
    return fig


def create_league_overview(df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top Performers by Avg Score', 'Team Skill Comparison', 'League Role Distribution', 'Performance vs Consistency'),
        specs=[[{"type": "bar"}, {"type": "polar"}], [{"type": "pie"}, {"type": "scatter"}]]
    )
    top_players = df.groupby('Player_ID')['Overall_Performance'].mean().nlargest(10)
    fig.add_trace(go.Bar(x=top_players.index, y=top_players.values, name='Top Performers', marker_color='#FF6B6B'), row=1, col=1)
    
    teams = df['Team'].unique()[:5]
    colors = px.colors.qualitative.Plotly
    stats_radar = ['Hits', 'Throws', 'Blocks', 'Dodges', 'Catches']
    for i, team in enumerate(teams):
        team_stats = df[df['Team'] == team][stats_radar].mean()
        fig.add_trace(go.Scatterpolar(r=team_stats.values, theta=stats_radar, fill='toself', name=team, line_color=colors[i]), row=1, col=2)
    
    if 'Player_Role' in df.columns:
        role_counts = df['Player_Role'].dropna().value_counts()
        if not role_counts.empty:
            fig.add_trace(go.Pie(labels=role_counts.index, values=role_counts.values, name="Roles"), row=2, col=1)
    
    player_summary = df.groupby('Player_ID').agg(Overall_Performance=('Overall_Performance', 'mean'), Win_Rate=('Win_Rate', 'first'), Consistency_Score=('Consistency_Score', 'first')).reset_index().dropna()
    fig.add_trace(go.Scatter(x=player_summary['Overall_Performance'], y=player_summary['Consistency_Score'], mode='markers', text=player_summary['Player_ID'], marker=dict(size=player_summary['Win_Rate'] * 20 + 5, color=player_summary['Win_Rate'], colorscale='Viridis', showscale=True, colorbar_title='Win Rate'), name='Performance vs Consistency'), row=2, col=2)
    fig.update_layout(height=800, title_text="League Overview Dashboard", polar=dict(radialaxis=dict(visible=True, range=[0, df[stats_radar].max().max()])))
    return fig


def create_specialization_analysis(df):
    st.header("Player Specialization Analysis")
    st.write("This section identifies players who are specialists in key skills by comparing their performance against the league average.")
    spec_stats = ['Hits', 'Throws', 'Dodges', 'Catches', 'Hit_Accuracy', 'Defensive_Efficiency', 'K/D_Ratio']
    player_avg_stats = df.groupby('Player_ID')[spec_stats].mean()
    league_avg = player_avg_stats.mean()
    specialization = player_avg_stats / (league_avg + 1e-6)
    top_specialized_players = specialization.std(axis=1).nlargest(20).index
    specialization_subset = specialization.loc[top_specialized_players]
    fig = px.imshow(specialization_subset, text_auto=".2f", aspect="auto", color_continuous_scale='Viridis', labels=dict(x="Statistic", y="Player", color="Specialization Score (x League Avg)"), title="Player Specialization Heatmap (vs. League Average)")
    fig.update_xaxes(side="top")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 A score of 2.0 means a player is twice as good as the average in that skill.")
    st.subheader("Top Specialists by Key Skill")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🛡️ **Top Catchers**")
        top_catchers = specialization.sort_values('Catches', ascending=False).head(5)
        st.dataframe(top_catchers[['Catches']].style.format("{:.2f}x Avg").background_gradient(cmap='Greens'))
    with col2:
        st.write("🎯 **Top Sharpshooters (Hit Accuracy)**")
        top_accurate = specialization.sort_values('Hit_Accuracy', ascending=False).head(5)
        st.dataframe(top_accurate[['Hit_Accuracy']].style.format("{:.2f}x Avg").background_gradient(cmap='Reds'))
    with col3:
        st.write("⚡ **Top 'Clutch' Players (K/D Ratio)**")
        top_kd = specialization.sort_values('K/D_Ratio', ascending=False).head(5)
        st.dataframe(top_kd[['K/D_Ratio']].style.format("{:.2f}x Avg").background_gradient(cmap='Purples'))


def generate_player_coaching_report(df, player_id):
    player_data = df[df['Player_ID'] == player_id]
    if player_data.empty:
        return ["No data for this player."], None

    player_avg_stats = player_data.mean(numeric_only=True)
    
    if 'Player_Role' not in player_data.columns or player_data['Player_Role'].isnull().all():
        player_role = "N/A"
        role_avg_stats = pd.Series()
    else:
        player_role = player_data['Player_Role'].iloc[0]
        role_avg_stats = df[df['Player_Role'] == player_role].mean(numeric_only=True)

    league_avg_stats = df.mean(numeric_only=True)
    stats_to_compare = ['Hits', 'Throws', 'Catches', 'Dodges', 'Hit_Accuracy', 'Defensive_Efficiency', 'K/D_Ratio']
    
    role_weaknesses = {}
    if not role_avg_stats.empty:
        role_weaknesses = {stat: (player_avg_stats.get(stat, 0) - role_avg_stats.get(stat, 0)) / (role_avg_stats.get(stat, 0) + 1e-6) for stat in stats_to_compare}
        role_weaknesses = {k: v for k, v in role_weaknesses.items() if v < -0.1}

    overall_weaknesses = {stat: (player_avg_stats.get(stat, 0) - league_avg_stats.get(stat, 0)) / (league_avg_stats.get(stat, 0) + 1e-6) for stat in stats_to_compare}
    overall_weaknesses = {k: v for k, v in overall_weaknesses.items() if v < -0.1}

    report = [f"### Coaching Focus for {player_id} ({player_role})"]
    advice_map = {
        'Hit_Accuracy': "🎯 **Suggestion**: Focus on throwing drills.",
        'Defensive_Efficiency': "🙌 **Suggestion**: Improve decision-making when targeted.",
        'Catches': "🛡️ **Suggestion**: Improve positioning and anticipation.",
        'Dodges': "🏃 **Suggestion**: Enhance agility and footwork.",
        'Hits': "💥 **Suggestion**: Be more aggressive offensively.",
        'K/D_Ratio': "⚡ **Suggestion**: Focus on survivability."
    }
    if not role_weaknesses and not overall_weaknesses:
        report.append("✅ **Well-Rounded Performer**: This player is performing at or above average.")
    if role_weaknesses:
        stat = min(role_weaknesses, key=role_weaknesses.get)
        report.append(f"**Role-Specific Weakness**: **{stat}**.")
        report.append(advice_map.get(stat))
    if overall_weaknesses:
        stat = min(overall_weaknesses, key=overall_weaknesses.get)
        report.append(f"**Overall Weakness**: **{stat}**.")
        if not role_weaknesses or stat != min(role_weaknesses, key=role_weaknesses.get):
             report.append(advice_map.get(stat))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f'{player_id} (You)', x=stats_to_compare, y=[player_avg_stats.get(s, 0) for s in stats_to_compare], marker_color='#FF6B6B'))
    if not role_avg_stats.empty:
        fig.add_trace(go.Bar(name='Role Average', x=stats_to_compare, y=[role_avg_stats.get(s, 0) for s in stats_to_compare], marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(name='League Average', x=stats_to_compare, y=[league_avg_stats.get(s, 0) for s in stats_to_compare], marker_color='#45B7D1'))
    
    fig.update_layout(barmode='group', title_text='Performance Comparison', xaxis_title="Statistic", yaxis_title="Average Value")
    return report, fig


def generate_team_coaching_report(df, team_id):
    team_data = df[df['Team'] == team_id]
    if team_data.empty:
        return ["No data for this team."]
    team_avg_stats = team_data.mean(numeric_only=True)
    league_avg_stats = df[df['Team'] != team_id].mean(numeric_only=True)
    stats_to_compare = ['Hits', 'Throws', 'Catches', 'Dodges', 'Hit_Accuracy', 'Defensive_Efficiency', 'Overall_Performance', 'K/D_Ratio']
    weaknesses = {stat: (team_avg_stats.get(stat, 0) - league_avg_stats.get(stat, 0)) / (league_avg_stats.get(stat, 0) + 1e-6) for stat in stats_to_compare}
    biggest_weakness = min(weaknesses, key=weaknesses.get)
    report = [f"### Coaching Focus for {team_id}", f"**Biggest Team Weakness**: The team's **{biggest_weakness}** is the furthest below the league average."]
    advice_map = {
        'Hit_Accuracy': "🎯 **Team Focus**: Dedicate a session to throwing accuracy.",
        'Defensive_Efficiency': "🙌 **Team Focus**: Run drills that simulate 2-on-1 situations.",
        'Catches': "🛡️ **Team Focus**: Emphasize the value of catching.",
        'Dodges': "🏃 **Team Focus**: A full-team agility session could be beneficial.",
        'Overall_Performance': "📈 **Team Focus**: Go back to basics.",
        'K/D_Ratio': "⚡ **Team Focus**: The team needs to improve its elimination efficiency."
    }
    report.append(advice_map.get(biggest_weakness, "Focus on improving this area through targeted drills."))
    
    if 'Player_Role' in team_data.columns and not team_data['Player_Role'].isnull().all():
        has_catcher = any('Catcher' in str(role) for role in team_data['Player_Role'].unique())
        if not has_catcher:
            report.append("\n**Strategic Gap**: The team lacks a dedicated 'Catcher' type player.")
    return report


def generate_insights(df):
    insights = []
    top_performer = df.groupby('Player_ID')['Overall_Performance'].mean().idxmax()
    top_score = df.groupby('Player_ID')['Overall_Performance'].mean().max()
    insights.append(f"🏆 **Top Performer**: {top_performer} with an average performance score of {top_score:.2f}")

    top_kd_player = df.groupby('Player_ID')['K/D_Ratio'].mean().idxmax()
    top_kd = df.groupby('Player_ID')['K/D_Ratio'].mean().max()
    insights.append(f"⚡ **Most Efficient Player**: {top_kd_player} is the most efficient eliminator with an incredible K/D Ratio of {top_kd:.2f}.")

    best_team = df.groupby('Team')['Overall_Performance'].mean().idxmax()
    insights.append(f"🥇 **Strongest Team**: {best_team} with the highest average performance.")
    
    if 'Player_Role' in df.columns and not df['Player_Role'].isnull().all():
        best_role = df.groupby('Player_Role')['Overall_Performance'].mean().idxmax()
        insights.append(f"✨ **Most Effective Role**: Players classified as **{best_role}** show the highest average performance across the league.")
    return insights