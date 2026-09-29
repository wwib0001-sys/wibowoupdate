import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, date, time, timezone, timedelta

# Auto-refresh helper (only triggers in live mode)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# --- 1. PAGE CONFIGURATION & DARK-BLUE STYLING ---
st.set_page_config(
    page_title="HVAC Monitoring Dashboard based on Occupancy Prediction", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Dark-blue application background */
    .stApp {
        background-color: #0a1128 !important;
        color: #f1f5f9 !important;
    }
    
    /* Completely hide sidebar */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 96%;
    }
    
    /* Control Toolbar Card */
    .control-card {
        background-color: #121e3a;
        border: 1px solid #1e325c;
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    
    /* Dark-blue Metric Cards */
    .metric-card {
        background-color: #121e3a;
        border: 1px solid #1e325c;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-title { 
        color: #94a3b8; 
        font-size: 12px; 
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 0.8px;
        margin-bottom: 6px; 
    }
    .metric-value { 
        color: #ffffff; 
        font-size: 30px; 
        font-weight: bold; 
    }
    .metric-subtext { 
        color: #38bdf8; 
        font-size: 14px; 
        font-weight: 600; 
        margin-top: 4px; 
    }
    
    /* Header Clock Banner Card */
    .clock-card {
        background-color: #121e3a;
        border: 1px solid #1e325c;
        border-radius: 12px;
        padding: 14px 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
        font-size: 24px;
    }
    .circle-indicator { 
        width: 18px; 
        height: 18px; 
        border-radius: 50%; 
        display: inline-block;
    }
    .circle-on { 
        background: radial-gradient(circle at 30% 30%, #4ade80, #16a34a); 
        box-shadow: 0 0 12px rgba(74, 222, 128, 0.6); 
    }
    .circle-off { 
        background: radial-gradient(circle at 30% 30%, #94a3b8, #475569); 
    }
    
    h1, h2, h3, h4, p {
        color: #ffffff !important;
    }
    .sub-description {
        color: #94a3b8 !important;
        font-size: 15px;
        margin-top: -8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. AUTOMATIC LOOPING SCROLLER SCRIPT ---
components.html("""
<script>
    const parentWin = window.parent;
    let scrollSpeed = 1;
    let intervalTime = 30;
    let isPaused = false;

    function autoScroll() {
        if (!isPaused && parentWin) {
            let maxScroll = parentWin.document.documentElement.scrollHeight - parentWin.innerHeight;
            let currentScroll = parentWin.scrollY || parentWin.pageYOffset;

            if (currentScroll >= maxScroll - 2) {
                isPaused = true;
                setTimeout(() => {
                    parentWin.scrollTo({ top: 0, behavior: 'smooth' });
                    setTimeout(() => {
                        isPaused = false;
                    }, 2500);
                }, 2000);
            } else {
                parentWin.scrollBy(0, scrollSpeed);
            }
        }
    }
    setInterval(autoScroll, intervalTime);
</script>
""", height=0, width=0)

# --- 3. DATA LOADING DIRECTLY FROM HVAC CSV ---
@st.cache_data(ttl=60)
def load_data():
    try:
        df = pd.read_csv('hvac_comparison.csv')
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    df = df.sort_index()

    # Map standard columns
    df['occ_true'] = df['occupancy_now']
    df['occ_pred'] = df['occupancy_forecast_15min']
    df['occ_actual_15m'] = df['occupancy_actual_15min_later']
    df['T_pred'] = df['predictive_room_temperature_C']
    df['power_pred'] = df['predictive_hvac_power_kW']
    
    # Calculate interval energy from cumulative
    df['energy_interval_pred'] = df['predictive_energy_cumulative_kWh'].diff().fillna(0).clip(lower=0)
    
    return df

df_full = load_data()
available_dates = df_full.index.normalize().unique().date

# --- 4. HEADER: TITLE & CLOCK BANNER ---
col_head_left, col_head_right = st.columns([2.3, 1.1])

with col_head_left:
    st.title("🏢 Smart HVAC Monitoring Dashboard based on Occupancy Prediction")
    st.markdown("<div class='sub-description'>Real-time facility environmental monitoring, equipment control state, and predictive demand tracking.</div>", unsafe_allow_html=True)

# Live Melbourne current time
try:
    import pytz
    melbourne_tz = pytz.timezone('Australia/Melbourne')
    now_melbourne = datetime.now(melbourne_tz)
except Exception:
    melbourne_tz = timezone(timedelta(hours=10))
    now_melbourne = datetime.now(melbourne_tz)

real_live_time = now_melbourne.time()

# --- 5. CONTROL TOOLBAR: LIVE VS. HISTORICAL FILTER ---
st.markdown("<div class='control-card'>", unsafe_allow_html=True)
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1.2, 2])

with col_ctrl1:
    is_live_mode = st.toggle("🔴 Real-Time Live Mode", value=True, help="Toggle OFF to inspect historical dates and filter system time.")

if is_live_mode:
    if st_autorefresh:
        st_autorefresh(interval=30000, key="live_refresh")

    sep29_dates = [d for d in available_dates if d.month == 9 and d.day == 29]
    selected_date = sep29_dates[0] if sep29_dates else available_dates[-1]
    active_time = real_live_time
    mode_label = "LIVE SYSTEM CLOCK"
    clock_color = "#ef4444"

    with col_ctrl2:
        st.write(f"📅 **Tracking Date:** {selected_date.strftime('%Y-%m-%d')}")
    with col_ctrl3:
        st.write(f"⏱️ **Tracking Mode:** Dynamic Real-Time Clock Sync ({active_time.strftime('%H:%M:%S')})")
else:
    mode_label = "INSPECTION MODE"
    clock_color = "#38bdf8"
    
    with col_ctrl2:
        selected_date = st.selectbox(
            "📅 Select Filter Date:", 
            options=available_dates, 
            index=len(available_dates)-1
        )
    
    df_day_candidates = df_full[df_full.index.date == selected_date]
    day_times = df_day_candidates.index.time
    
    with col_ctrl3:
        if len(day_times) > 0:
            active_time = st.select_slider("⏱️ Scrub Inspection Time:", options=day_times, value=day_times[-1])
        else:
            active_time = time(23, 59)

st.markdown("</div>", unsafe_allow_html=True)

# Render clock banner matching selected mode
with col_head_right:
    st.markdown(f"""
    <div class="clock-card">
        <div>
            <div style="font-size: 11px; font-weight: bold; color: {clock_color}; display: flex; align-items: center; gap: 6px; letter-spacing: 0.5px;">
                <span style="height: 8px; width: 8px; background-color: {clock_color}; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px {clock_color};"></span>
                {mode_label}
            </div>
            <div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 3px;">
                {selected_date.strftime('%A, %b %d, %Y')}
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">MELBOURNE TIME</div>
            <div style="font-size: 26px; font-weight: bold; color: #38bdf8; line-height: 1.1;">
                {active_time.strftime('%H:%M:%S')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 6. DATA FILTERING & CORE METRIC VALUES ---
df_day = df_full[df_full.index.date == selected_date].copy()
df_live = df_day[df_day.index.time <= active_time]

if df_live.empty:
    df_live = df_day.iloc[:1]

current_temp = df_live['T_pred'].iloc[-1]
current_power = df_live['power_pred'].iloc[-1]
current_occ = int(df_live['occ_true'].iloc[-1])
forecast_occ = int(df_live['occ_pred'].iloc[-1])
total_energy_today = df_live['energy_interval_pred'].sum()

# Compute Realisation Statistics (Day-to-Date)
eval_subset = df_live.dropna(subset=['occ_pred', 'occ_actual_15m'])
if len(eval_subset) > 0:
    mae_val = np.mean(np.abs(eval_subset['occ_actual_15m'] - eval_subset['occ_pred']))
    rmse_val = np.sqrt(np.mean((eval_subset['occ_actual_15m'] - eval_subset['occ_pred'])**2))
    bin_actual = (eval_subset['occ_actual_15m'] > 0).astype(int)
    bin_pred = (eval_subset['occ_pred'] > 0).astype(int)
    bin_acc = np.mean(bin_actual == bin_pred) * 100
else:
    mae_val, rmse_val, bin_acc = 0.0, 0.0, 100.0

if current_power > 0:
    hvac_status_html = "<span class='status-badge' style='color: #4ade80;'><span class='circle-indicator circle-on'></span> RUNNING</span>"
else:
    hvac_status_html = "<span class='status-badge' style='color: #94a3b8;'><span class='circle-indicator circle-off'></span> STANDBY</span>"

# --- 7. TOP KPI CARDS ---
col1, col2, col3, col4, col5 = st.columns([1.1, 1, 1, 1.2, 1.2])

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Device Status</div>
        <div class="metric-value">{hvac_status_html}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Indoor Temp</div>
        <div class="metric-value">{current_temp:.1f} °C</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Energy</div>
        <div class="metric-value">{total_energy_today:.1f} kWh</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Occupancy Demand</div>
        <div class="metric-value">{current_occ} <span style="font-size: 15px; font-weight: normal; color: #94a3b8;">Now</span></div>
        <div class="metric-subtext">📈 Forecast (15m): {forecast_occ}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Prediction Realisation</div>
        <div class="metric-value">{bin_acc:.1f}% <span style="font-size: 14px; font-weight: normal; color: #94a3b8;">State Acc</span></div>
        <div class="metric-subtext">MAE: {mae_val:.2f} | RMSE: {rmse_val:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 8. REUSABLE PLOT HELPER ---
def create_dark_blue_plot(data, y_col, title, y_label, line_color, is_area=False, is_step=False, hlines=None):
    fig = go.Figure()
    
    if not data.empty and y_col in data.columns:
        if is_area:
            r = int(line_color.lstrip("#")[0:2], 16)
            g = int(line_color.lstrip("#")[2:4], 16)
            b = int(line_color.lstrip("#")[4:6], 16)
            fill_color = f'rgba({r}, {g}, {b}, 0.2)'
            fig.add_trace(go.Scatter(
                x=data.index, y=data[y_col], 
                mode='lines', 
                line=dict(color=line_color, width=2.5), 
                fill='tozeroy', 
                fillcolor=fill_color, 
                name=y_label
            ))
        elif is_step:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[y_col], 
                mode='lines', 
                line=dict(color=line_color, width=2.5), 
                line_shape='hv', 
                name=y_label
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[y_col], 
                mode='lines', 
                line=dict(color=line_color, width=2.5), 
                name=y_label
            ))
    
    if hlines:
        for hline in hlines:
            fig.add_hline(y=hline, line_dash="dash", line_color="#ef4444", opacity=0.7)

    marker_dt = datetime.combine(selected_date, active_time)
    marker_text = "LIVE" if is_live_mode else "INSPECT"
    marker_color = "#ef4444" if is_live_mode else "#38bdf8"

    fig.add_vline(
        x=marker_dt, 
        line_width=2, 
        line_dash="solid", 
        line_color=marker_color, 
        annotation_text=marker_text, 
        annotation_position="top right",
        annotation_font_color=marker_color
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f1f5f9")),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="", 
        yaxis_title=y_label,
        hovermode="x unified",
        plot_bgcolor="#121e3a", 
        paper_bgcolor="#121e3a",
        font=dict(color="#94a3b8"),
        xaxis=dict(
            showgrid=True, 
            gridcolor="#1e325c", 
            zeroline=False,
            range=[datetime.combine(selected_date, time.min), datetime.combine(selected_date, time.max)]
        ),
        yaxis=dict(showgrid=True, gridcolor="#1e325c", zeroline=False),
        legend=dict(font=dict(color="#f1f5f9"))
    )
    return fig

# --- 9. CHARTS ---
# Row 1: Temperature Profile
st.subheader("🌡️ Temperature Profile")
fig_temp = create_dark_blue_plot(df_live, 'T_pred', "Indoor Temperature Tracking", "Temperature (°C)", "#38bdf8", hlines=[22.8, 25.8])
fig_temp.add_hrect(
    y0=22.8, y1=25.8, 
    line_width=0, 
    fillcolor="#22c55e", 
    opacity=0.15, 
    annotation_text="Comfort Band (22.8 - 25.8°C)", 
    annotation_position="top left",
    annotation_font_color="#4ade80"
)
st.plotly_chart(fig_temp, use_container_width=True)

# Row 2: Occupancy Realisation vs. Power Draw
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.subheader("👥 Occupancy Realisation (15m Ahead)")
    fig_realisation = go.Figure()
    
    if not df_live.empty:
        # Ground Truth Realised (15 min later)
        if 'occ_actual_15m' in df_live.columns:
            fig_realisation.add_trace(go.Scatter(
                x=df_live.index, 
                y=df_live['occ_actual_15m'], 
                mode='lines', 
                name='Actual Realised (15m Later)',
                line=dict(color='#4ade80', width=2.5),
                line_shape='hv'
            ))
        
        # Model 15-minute prediction
        fig_realisation.add_trace(go.Scatter(
            x=df_live.index, 
            y=df_live['occ_pred'], 
            mode='lines', 
            name='Model Forecast (15m Ahead)',
            line=dict(color='#38bdf8', width=2, dash='dot'),
            line_shape='hv'
        ))
        
        # Current occupancy now
        fig_realisation.add_trace(go.Scatter(
            x=df_live.index, 
            y=df_live['occ_true'], 
            mode='lines', 
            name='Occupancy (Now)',
            line=dict(color='#94a3b8', width=1.5, dash='dash'),
            line_shape='hv'
        ))

    marker_dt = datetime.combine(selected_date, active_time)
    marker_text = "LIVE" if is_live_mode else "INSPECT"
    marker_color = "#ef4444" if is_live_mode else "#38bdf8"
    fig_realisation.add_vline(x=marker_dt, line_width=2, line_dash="solid", line_color=marker_color)

    fig_realisation.update_layout(
        title=dict(text="Forecast vs. Ground Truth Realisation", font=dict(size=16, color="#f1f5f9")),
        xaxis_title="", yaxis_title="Occupants",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#121e3a", paper_bgcolor="#121e3a",
        font=dict(color="#94a3b8"),
        xaxis=dict(
            showgrid=True, gridcolor="#1e325c", zeroline=False,
            range=[datetime.combine(selected_date, time.min), datetime.combine(selected_date, time.max)]
        ),
        yaxis=dict(showgrid=True, gridcolor="#1e325c", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f1f5f9"))
    )
    st.plotly_chart(fig_realisation, use_container_width=True)

with col_c2:
    st.subheader("⚡ Energy Usage (HVAC Demand)")
    fig_energy = create_dark_blue_plot(df_live, 'power_pred', "HVAC Power Draw (Demand)", "Power (kW)", "#fb923c", is_area=True)
    st.plotly_chart(fig_energy, use_container_width=True)
