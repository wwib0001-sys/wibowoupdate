import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("🏢 Occupancy‑Driven HVAC Backtest (Constant Outdoor Temp)")

@st.cache_data
def load_data():
    # Try to load CSV
    try:
        df = pd.read_csv('backtest_hvac_constant_temp.csv')
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        raise

    # Convert timestamp to datetime and set as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    else:
        # Use first column as index
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.set_index(first_col)

    # Sort index
    df = df.sort_index()

    # REMOVED: df = df.dropna(subset=[df.index.name])
    # (This was causing the KeyError)

    # Map standard names to the columns present in your CSV
    # Mapping for easy access
    df['occ_true'] = df['occupancy_actual']
    df['occ_pred'] = df['occupancy_predicted']
    
    # Temperature: There is no separate predicted temperature in the CSV.
    # We will use the actual room temperature for both, or you can replace
    # with a simulated column if you add one later.
    df['T_true'] = df['room_temperature_C']
    df['T_pred'] = df['room_temperature_C']  # Assuming no pred, using actual
    
    # Energy: Use the cumulative energy column directly (Actual)
    # If you have a predicted cumulative column, use that; otherwise, use the same.
    if 'hvac_energy_cumulative_kWh' in df.columns:
        df['energy_true'] = df['hvac_energy_cumulative_kWh']
        # For now, use actual for predicted as well. If you have a separate pred column, map it here.
        df['energy_pred'] = df['hvac_energy_cumulative_kWh']
    else:
        # Fallback: compute from interval if cumulative missing
        if 'hvac_energy_interval_kWh' in df.columns:
            df['energy_true'] = df['hvac_energy_interval_kWh'].cumsum()
            df['energy_pred'] = df['hvac_energy_interval_kWh'].cumsum()
        else:
            st.error("No energy columns found in CSV!")
            st.stop()

    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    [df.index.min().date(), df.index.max().date()]
)
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_filtered = df.loc[mask]
else:
    df_filtered = df

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    # Calculate RMSE between T_pred and T_true (they are the same now)
    rmse_temp = np.sqrt(np.mean((df_filtered['T_pred'] - df_filtered['T_true'])**2))
    st.metric("Temperature RMSE", f"{rmse_temp:.2f} °C")
with col2:
    # Energy difference: Since we are using actual for both, this will be 0. Change if you have a true pred column.
    energy_diff = df_filtered['energy_pred'].iloc[-1] - df_filtered['energy_true'].iloc[-1]
    st.metric("Energy Difference (Pred - True)", f"{energy_diff:.2f} kWh")
with col3:
    # Comfort violation for T_true (since T_pred is the same)
    comfort_true = ((df_filtered['T_true'] < 20) | (df_filtered['T_true'] > 26)).mean()
    comfort_pred = ((df_filtered['T_pred'] < 20) | (df_filtered['T_pred'] > 26)).mean()
    st.metric("Comfort Violation (True)", f"{comfort_true*100:.1f}%")
    st.metric("Comfort Violation (Pred)", f"{comfort_pred*100:.1f}%")

# Occupancy plot
fig_occ = go.Figure()
fig_occ.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['occ_true'], 
                             name='Actual Occupancy', line=dict(color='blue')))
fig_occ.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['occ_pred'], 
                             name='Predicted Occupancy', line=dict(color='red', dash='dash')))
fig_occ.update_layout(title="Occupancy Over Time", xaxis_title="Time", yaxis_title="Occupancy")

# Temperature plot
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['T_true'], 
                              name='Actual Temp', line=dict(color='green')))
fig_temp.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['T_pred'], 
                              name='Predicted Temp', line=dict(color='orange', dash='dash')))
fig_temp.add_hline(y=20, line_dash="dot", line_color="gray", annotation_text="Comfort lower")
fig_temp.add_hline(y=26, line_dash="dot", line_color="gray", annotation_text="Comfort upper")
fig_temp.update_layout(title="Indoor Temperature", xaxis_title="Time", yaxis_title="Temperature (°C)")

# Energy plot
fig_energy = go.Figure()
fig_energy.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['energy_true'], 
                                name='Actual Energy', line=dict(color='purple')))
fig_energy.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['energy_pred'], 
                                name='Predicted Energy', line=dict(color='brown', dash='dash')))
fig_energy.update_layout(title="Cumulative HVAC Energy", xaxis_title="Time", yaxis_title="Energy (kWh)")

# Show plots
st.plotly_chart(fig_occ, use_container_width=True)
st.plotly_chart(fig_temp, use_container_width=True)
st.plotly_chart(fig_energy, use_container_width=True)

# Optional: scatter of predicted vs actual energy
st.subheader("Energy Scatter")
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=df_filtered['energy_true'], y=df_filtered['energy_pred'], 
                                 mode='markers', marker=dict(color='darkblue')))
fig_scatter.add_trace(go.Scatter(x=[df_filtered['energy_true'].min(), df_filtered['energy_true'].max()],
                                 y=[df_filtered['energy_true'].min(), df_filtered['energy_true'].max()],
                                 mode='lines', name='Ideal', line=dict(dash='dash', color='red')))
fig_scatter.update_layout(title="Predicted vs Actual Cumulative Energy", 
                          xaxis_title="Actual Energy (kWh)", yaxis_title="Predicted Energy (kWh)")
st.plotly_chart(fig_scatter, use_container_width=True)
