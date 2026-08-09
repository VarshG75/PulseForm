import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import pickle # NEW: We need this to open the AI brain!

# Set up the page style
st.set_page_config(page_title="PulseForm", layout="centered")

# The Dashboard UI
st.title("🫀 PulseForm: Cardiovascular Screener")
st.write("Welcome to the PulseForm prototype. Please upload a 15-second video of your finger covering the camera and flash.")

# File uploader
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov"])

# The "Integration" button
if uploaded_video is not None:
    st.success("Video uploaded successfully!")
    
    if st.button("Analyze Vascular Health"):
        with st.spinner("Extracting pulse wave and asking the AI..."):
            time.sleep(2) # Fake loading time
            
        st.subheader("📊 Extracted Pulse Wave:")
        
        # Load the hospital data as a placeholder
        df = pd.read_csv("app_and_ml/data/PPG_Dataset.csv", header=None)
        patient_wave = df.iloc[1, :1000] # Grabbing Patient 1
        
        # Build the Plotly graph
        fig = go.Figure(data=go.Scatter(y=patient_wave, mode='lines', line=dict(color='#ff4b4b', width=2)))
        fig.update_layout(
            xaxis_title="Time (Milliseconds)",
            yaxis_title="Blood Volume",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- NEW AI INTEGRATION ---
        # 1. Open the saved AI brain
        with open("app_and_ml/vascular_model.pkl", "rb") as f:
            ai_model = pickle.load(f)
            
        # 2. Reformat the single wave so the AI can read it (it expects a 2D grid)
        wave_for_ai = patient_wave.values.reshape(1, -1)
        
        # 3. Ask the AI to predict the age!
        predicted_age = ai_model.predict(wave_for_ai)[0]
        # --------------------------
        
        st.subheader("🩺 Estimated Results:")
        st.write("**Heart Rate:** 72 BPM") # (Shreyas will calculate this later)
        st.write("**Stiffness Index:** 8.4") # (Shreyas will calculate this later)
        
        # Notice we replaced the fake "45" with the AI's actual prediction!
        st.write(f"**Vascular Age:** {int(predicted_age)} years")