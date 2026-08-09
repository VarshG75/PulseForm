import pandas as pd
import plotly.graph_objects as go

print("1. Pandas is opening the massive dataset... please wait.")

# Open the CSV file
df = pd.read_csv("app_and_ml/data/PPG_Dataset.csv", header=None)

print(f"2. Success! Pandas loaded {len(df)} patients into memory.")

# Isolate Patient 1 (Row 0) and their first 1000 milliseconds of data
patient_wave = df.iloc[3, :1000] 

print("3. Drawing the heartbeat graph...")

# Draw the graph using Plotly
fig = go.Figure(data=go.Scatter(y=patient_wave, mode='lines', line=dict(color='#ff4b4b', width=2)))

fig.update_layout(
    title="Patient 1: Raw Photoplethysmography (PPG) Wave",
    xaxis_title="Time (Milliseconds)",
    yaxis_title="Blood Volume (Light Reflection)",
    template="plotly_dark"
)

fig.show()