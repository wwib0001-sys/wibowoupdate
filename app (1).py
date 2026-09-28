import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta, time

# Configure page for Master Control layout
st.set_page_config(page_title="GEB Master Control", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
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

    # Map standard column names based on the hvac_comparison.csv schema
    df['occ_true'] = df['occupancy_now']
    df['occ_pred_15m'] = df['occupancy_forecast_15min']
    df['occ_actual_15m'] = df['occupancy_actual_15min_later']
    df['T_true'] = df['scheduled_room_temperature_C']
    df['T_pred'] = df['predictive_room_temperature_C']
    df['power_true'] = df['scheduled_hvac_power_kW']
    df['power_pred_base'] = df['predictive_hvac_power_kW']
    
    # Derive interval energy from cumulative columns dynamically
    df['energy_interval_true'] = df['scheduled_energy_cumulative_kWh'].diff().fillna(0).clip(lower=0)
    df['energy_interval_pred_base'] = df['predictive_energy_cumulative_kWh'].diff().fillna(0).clip(lower=0)
    
    return df

df_full = load_data()
unique_dates = df_full.index.normalize().unique().date

# --- DISPATCH CONSOLE (SIDEBAR) ---
st.sidebar.markdown("## 🎛️ DISPATCH CONSOLE")
if len(unique_dates) == 0:
    st.error("No valid dates found in the dataset.")
    st.stop()

selected_day = st.sidebar.selectbox("📅 Select Operation Date", options=unique_dates)
df_day = df_full[df_full.index.date == selected_day]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔴 LIVE CLOCK FEED")
if not df_day.empty:
    time_options = df_day.index.time
    current_time = st.sidebar.select_slider("System Time", options=time_options, value=time_options[len(time_options)//2])
else:
    current_time = time(12, 0)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ TRANSMISSION INSTRUCTIONS")
event_type = st.sidebar.selectbox("Grid Event Type", options=["Contingency FCAS (15m)", "Peak Demand / RERT (2h)"])
issue_load_shed = st.sidebar.toggle("🚨 INITIATE LOAD SHED", value=False)
shed_duration_mins = 15 if "FCAS" in event_type else 120

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ LOCAL SAFETY OVERRIDES")
max_temp_override = st.sidebar.slider("Critical Temp Limit (°C)", min_value=26.0, max_value=30.0, value=27.5, step=0.1, help="Auto-aborts grid shed if exceeded.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 TARIFF PARAMETERS")
energy_rate = st.sidebar.number_input("Current Rate ($/kWh)", value=0.35, step=0.01)

# --- GRID OVERRIDE ENGINE ---
if issue_load_shed:
    event_start_time = current_time
    event_end_dt = datetime.combine(selected_day, current_time) + timedelta(minutes=shed_duration_mins)
    event_end_time = event_end_dt.time()
else:
    event_start_time = None
    event_end_time = None

def apply_grid_override(df_day_subset):
    df_dyn = df_day_subset.copy()
    if df_dyn.empty:
        df_dyn['power_pred'] = []
        df_dyn['T_pred_dynamic'] = []
        df_dyn['safety_override_active'] = []
        df_dyn['energy_interval_pred'] = []
        return df_dyn

    df_dyn['power_pred'] = df_dyn['power_pred_base']
    df_dyn['T_pred_dynamic'] = df_dyn['T_pred']
    df_dyn['safety_override_active'] = False
    df_dyn['energy_interval_pred'] = df_dyn['energy_interval_pred_base']
    
    if not issue_load_shed or event_start_time is None:
        return df_dyn

    temp_drift = 0.0 
    for i in range(len(df_dyn)):
        t_stamp = df_dyn.index[i].time()
        
        if event_start_time <= event_end_time:
            is_grid_event = event_start_time <= t_stamp <= event_end_time
        else:
            is_grid_event = t_stamp >= event_start_time or t_stamp <= event_end_time
            
        current_temp_with_drift = df_dyn['T_pred'].iloc[i] + temp_drift

        if is_grid_event:
            if current_temp_with_drift >= max_temp_override:
                df_dyn.loc[df_dyn.index[i], 'safety_override_active'] = True
                temp_drift = 0.0 
            else:
                df_dyn.loc[df_dyn.index[i], 'power_pred'] = 0.0
                outdoor_t = df_dyn['outdoor_temperature_C'].iloc[i]
                if current_temp_with_drift < outdoor_t:
                    temp_drift += (outdoor_t - current_temp_with_drift) * 0.05 
        else:
            if temp_drift > 0: temp_drift *= 0.5 
            if temp_drift < 0.1: temp_drift = 0.0
                
        df_dyn.loc[df_dyn.index[i], 'T_pred_dynamic'] = df_dyn['T_pred'].iloc[i] + temp_drift

    df_dyn['energy_interval_pred'] = df_dyn['power_pred'] * (5/60) 
    return df_dyn

df_live = df_day[df_day.index.time <= current_time]
df_live_dyn = apply_grid_override(df_live)

# --- METRICS CALCULATIONS ---
daily_energy_true = df_live_dyn['energy_interval_true'].sum() if 'energy_interval_true' in df_live_dyn.columns else 0
daily_energy_pred = df_live_dyn['energy_interval_pred'].sum() if 'energy_interval_pred' in df_live_dyn.columns else 0
energy_diff = daily_energy_pred - daily_energy_true

def calc_live_metrics(data):
    if data.empty: return 1.0, 1.0, 0, False
    comf_true = data['scheduled_comfort_ok'].dropna().mean()
    data['pred_comfort_dyn_ok'] = np.where(data['occ_true'] > 0, (data['T_pred_dynamic'] >= 22.8) & (data['T_pred_dynamic'] <= 25.8), np.nan)
    comf_pred = data['pred_comfort_dyn_ok'].dropna().mean()
    if pd.isna(comf_true): comf_true = 1.0
    if pd.isna(comf_pred): comf_pred = 1.0
    
    load_reduction = 0
    override_triggered = data['safety_override_active'].any() if 'safety_override_active' in data.columns else False
    
    if issue_load_shed and event_start_time and event_end_time:
        if event_start_time <= event_end_time:
            event_mask = (data.index.time >= event_start_time) & (data.index.time <= event_end_time)
        else:
            event_mask = (data.index.time >= event_start_time) | (data.index.time <= event_end_time)
        event_data = data[event_mask]
        if not event_data.empty and 'power_pred' in event_data.columns:
            load_reduction = event_data['power_true'].mean() - event_data['power_pred'].mean()

    return comf_true, comf_pred, load_reduction, override_triggered

comf_true, comf_pred, load_reduction, has_override = calc_live_metrics(df_live_dyn)
savings_usd = abs(energy_diff) * energy_rate if energy_diff < 0 else (energy_diff * energy_rate)

# ==========================================
# UI: MASTER CONTROL DASHBOARD
# ==========================================
st.title("⚡ ELECTRICAL MASTER CONTROL: GEB TERMINAL")

if has_override:
    st.error(f"🚨 CRITICAL ALARM: SAFETY OVERRIDE ACTIVE AT {current_time}. GRID INSTRUCTION ABORTED DUE TO THERMAL LIMIT.")
elif issue_load_shed and event_start_time <= current_time <= event_end_time:
    st.warning(f"⚠️ GRID DEMAND RESPONSE ACTIVE. SHEDDING LOAD UNTIL {event_end_time}.")
else:
    st.success(f"✅ SYSTEM NOMINAL. STANDBY MODE.")

st.markdown("### 🎛️ REAL-TIME TELEMETRY HUD")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("⚡ Firm Capacity Shed", f"{load_reduction:.2f} kW" if issue_load_shed else "0.00 kW")
with col_m2:
    label = "💸 Opex Variance (Daily)" if energy_diff <= 0 else "💸 Opex Overrun (Daily)"
    st.metric(label, f"${savings_usd:.2f}", delta=f"{energy_diff:.2f} kWh vs Sched", delta_color="inverse")
with col_m3:
    st.metric("Comfort Compliance (Base)", f"{comf_true*100:.1f}%")
with col_m4:
    delta_comf = (comf_pred - comf_true) * 100
    st.metric("Comfort Compliance (Predictive)", f"{comf_pred*100:.1f}%", delta=f"{delta_comf:.1f}% vs Base", delta_color="normal")

st.markdown("---")

# 3. MULTI-MONITOR CHART LAYOUT
def create_control_plot(data, y_true_col, y_pred_col, title, y_label, true_name, pred_name, hlines=[], show_grid=False, is_step=False):
    fig = go.Figure()
    
    if not data.empty and y_true_col in data.columns and y_pred_col in data.columns:
        if is_step:
            fig.add_trace(go.Scatter(x=data.index, y=data[y_true_col], name=true_name, line=dict(color='#00d2ff', width=2), mode='lines', line_shape='hv'))
            fig.add_trace(go.Scatter(x=data.index, y=data[y_pred_col], name=pred_name, line=dict(color='#ff2a2a', dash='dash', width=2), mode='lines', line_shape='hv'))
        else:
            fig.add_trace(go.Scatter(x=data.index, y=data[y_true_col], name=true_name, line=dict(color='#00d2ff', width=2)))
            fig.add_trace(go.Scatter(x=data.index, y=data[y_pred_col], name=pred_name, line=dict(color='#ff2a2a', dash='dash', width=2)))
        
        if 'safety_override_active' in data.columns and data['safety_override_active'].any():
            override_data = data[data['safety_override_active']]
            fig.add_trace(go.Scatter(x=override_data.index, y=override_data[y_pred_col], mode='markers', name='Override', marker=dict(color='red', size=8, symbol='x')))
        
    for hline in hlines:
        fig.add_hline(y=hline, line_dash="dot", line_color="gray")
        
    full_day_start = datetime.combine(selected_day, datetime.min.time())
    full_day_end = datetime.combine(selected_day, time(23, 59))
        
    if show_grid and issue_load_shed and event_start_time and event_end_time:
        event_start_dt = datetime.combine(selected_day, event_start_time)
        event_end_dt = datetime.combine(selected_day, event_end_time)
        if event_end_dt.time() < event_start_dt.time(): event_end_dt += timedelta(days=1)
        fig.add_vrect(x0=event_start_dt, x1=event_end_dt, fillcolor="orange", opacity=0.2, layer="below", line_width=0, annotation_text=f"GRID DISPATCH", annotation_position="top left")
        
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='white')),
        xaxis_title="", yaxis_title=y_label,
        hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20),
        xaxis_range=[full_day_start, full_day_end],
        plot_bgcolor="rgba(17,17,17,1)", paper_bgcolor="rgba(17,17,17,1)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
    return fig

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.plotly_chart(create_control_plot(df_live_dyn, 'power_true', 'power_pred', "ELECTRICAL LOAD (kW)", "Power (kW)", "Base Load", "Predictive Load", show_grid=True, is_step=True), use_container_width=True)
    st.plotly_chart(create_control_plot(df_live_dyn, 'occ_true', 'occ_pred_15m', "OCCUPANCY FORECAST (15m Ahead)", "Occupants", "Actual", "Forecasted", is_step=True), use_container_width=True)

with col_chart2:
    st.plotly_chart(create_control_plot(df_live_dyn, 'T_true', 'T_pred_dynamic', "THERMAL SAFETY ZONE (°C)", "Temp (°C)", "Base Temp", "Predictive Temp", hlines=[22.8, 25.8, max_temp_override], show_grid=True), use_container_width=True)
    
    # Calculate cumulative charts natively for plotting
    df_live_dyn['cum_energy_true'] = df_live_dyn['energy_interval_true'].cumsum()
    df_live_dyn['cum_energy_pred'] = df_live_dyn['energy_interval_pred'].cumsum()
    st.plotly_chart(create_control_plot(df_live_dyn, 'cum_energy_true', 'cum_energy_pred', "CUMULATIVE ENERGY (kWh)", "Energy (kWh)", "Base Target", "Predictive Actual", show_grid=True), use_container_width=True)
