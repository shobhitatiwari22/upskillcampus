import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go

# 1. PAGE LAYOUT & GROUNDWORK CONFIGURATION
st.set_page_config(
    page_title="AI SmartCity Traffic Hub",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. INJECT LUXURY GLASSMORPHIC INTERFACE STYLING
st.markdown("""
    <style>
        @import url('https://googleapis.com');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Glassmorphic Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            margin-bottom: 15px;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.8rem;
            color: #64ffda;
            font-weight: 700;
        }
        .metric-desc {
            font-size: 0.75rem;
            color: #a8b2d1;
            margin-top: 5px;
        }
        
        .section-title {
            font-weight: 700;
            color: #ccd6f6;
            margin-top: 20px;
            margin-bottom: 10px;
            border-left: 4px solid #64ffda;
            padding-left: 10px;
            font-size: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# 3. SECURELY LOAD DEPLOYED PRODUCTION ARTIFACTS
import os
@st.cache_resource
def load_production_model():
    base_dir = os.path.dirname(_file_)
    model_path = os.path.join(base_dir, 'traffic_production_model.pkl')
    return joblib.load(model_path)

try:
    payload = load_production_model()
    model = payload['model']
    features_list = payload['features']
except FileNotFoundError:
    st.error("❌ Model weight configuration framework 'traffic_production_model.pkl' not detected in directory array loop.")
    st.stop()

# 4. CONTROL PANEL SIDEBAR SELECTORS
st.sidebar.markdown("""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h2 style='color: #64ffda; margin-bottom: 0;'>HUB CONTROL</h2>
        <p style='color: #8892b0; font-size: 0.8rem;'>AI SmartCity Dispatcher</p>
    </div>
    <hr style='border-color: rgba(255,255,255,0.1); margin-top: 0;'>
""", unsafe_allow_html=True)

junction = st.sidebar.selectbox(
    "📍 Select Monitoring Station",
    options=[1, 2, 3, 4],
    format_func=lambda x: f"Junction {x} Control"
)

selected_date = st.sidebar.date_input(
    "📅 Select Target Forecast Horizon",
    datetime.date.today()
)

st.sidebar.markdown("---")

# 5. DYNAMIC SPATIAL-TEMPORAL CYCLICAL INFERENCE ENGINE
day_of_week = selected_date.weekday()
month = selected_date.month
is_weekend = 1 if day_of_week >= 5 else 0

timeline_records = []
for hr in range(24):
    timeline_records.append({
        'Junction': junction,
        'Hour': hr,
        'Is_Weekend': is_weekend,
        'Hour_sin': np.sin(2 * np.pi * hr / 24.0),
        'Hour_cos': np.cos(2 * np.pi * hr / 24.0),
        'Day_sin': np.sin(2 * np.pi * day_of_week / 7.0),
        'Day_cos': np.cos(2 * np.pi * day_of_week / 7.0),
        'Month_sin': np.sin(2 * np.pi * month / 12.0),
        'Month_cos': np.cos(2 * np.pi * month / 12.0),
        'Display_Hour': hr
    })

forecast_df = pd.DataFrame(timeline_records)
X_infer = forecast_df[features_list]

# Calculate prediction inferences via XGBoost matrix
forecast_df['Predicted_Vehicles'] = model.predict(X_infer)
forecast_df['Predicted_Vehicles'] = forecast_df['Predicted_Vehicles'].clip(lower=0).round().astype(int)

# 6. APP MAIN WORKSPACE DISPLAY HEADER
st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.05);'>
        <h1 style='margin:0; color:#ffffff; font-size:2.2rem;'>🏙️ Intelligent Grid Management Operations</h1>
        <p style='margin:5px 0 0 0; color:#94a3b8; font-size:1rem;'>Live predictive dispatch timelines for <b>Junction {junction}</b> on <b>{selected_date.strftime('%A, %B %d, %Y')}</b></p>
    </div>
""", unsafe_allow_html=True)

# 7. METRICS PRESENTATION PANEL (Beautiful CSS Columns)
m_col1, m_col2, m_col3 = st.columns(3)

peak_load = forecast_df['Predicted_Vehicles'].max()
peak_hour = forecast_df.loc[forecast_df['Predicted_Vehicles'].idxmax(), 'Display_Hour']
time_string = datetime.time(hour=peak_hour).strftime('%I:%M %p')
avg_flow = int(round(forecast_df['Predicted_Vehicles'].mean()))
daily_aggregate = int(forecast_df['Predicted_Vehicles'].sum())

with m_col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔥 Peak Congestion Spike</div>
            <div class="metric-value">{peak_load} <span style='font-size:1rem; color:#8892b0;'>vehicles</span></div>
            <div class="metric-desc">Forecasted spike boundary occurs at <b>{time_string}</b></div>
        </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ Average Daily Flow</div>
            <div class="metric-value">{avg_flow} <span style='font-size:1rem; color:#8892b0;'>units/hr</span></div>
            <div class="metric-desc">Steady baseline operational flow calculated across 24h</div>
        </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Volumetric Aggregate</div>
            <div class="metric-value">{daily_aggregate:,} <span style='font-size:1rem; color:#8892b0;'>total vehicles</span></div>
            <div class="metric-desc">Total footprints traversing this sector timeline loop</div>
        </div>
    """, unsafe_allow_html=True)

# 8. HIGH-END INTERACTIVE PLOTLY CHARTING WINDOW
st.markdown('<div class="section-title">🔮 Interactive Prediction Flow Sequence Timeline</div>', unsafe_allow_html=True)

time_labels = [f"{h:02d}:00" for h in forecast_df['Display_Hour']]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_labels,
    y=forecast_df['Predicted_Vehicles'],
    mode='lines+markers',
    line=dict(color='#64ffda', width=3),
    marker=dict(size=7, color='#0f172a', line=dict(color='#64ffda', width=2)),
    fill='tozeroy',
    fillcolor='rgba(100, 255, 218, 0.08)',
    hovertemplate='<b>Time:</b> %{x}<br><b>Forecast:</b> %{y} vehicles<extra></extra>'
))

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=10, b=20),
    height=380,
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        tickfont=dict(color='#8892b0')
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        tickfont=dict(color='#8892b0')
    ),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# 9. HORIZONTAL GRID LOG MATRIX
st.markdown('<div class="section-title">📋 Scheduled Operational Grid Logs</div>', unsafe_allow_html=True)

log_display = forecast_df[['Display_Hour', 'Predicted_Vehicles']].copy()
log_display['Display_Hour'] = log_display['Display_Hour'].apply(lambda h: f"{h:02d}:00")
log_display.columns = ['Operational Hour Slot', 'Predicted Traffic Count (Vehicles/Hour)']

st.dataframe(log_display.set_index('Operational Hour Slot').T, use_container_width=True)
