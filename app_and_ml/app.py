import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

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
        with st.spinner("Extracting pulse wave... (Processing data)"):
            time.sleep(2) # Fake loading time
            
        st.subheader("📊 Extracted Pulse Wave:")
        
        # --- NEW GRAPH CODE ---
        # We load the hospital data as a placeholder until Shreyas finishes the video extractor
        df = pd.read_csv("app_and_ml/data/PPG_Dataset.csv", header=None)
        patient_wave = df.iloc[1, :1000] # Grabbing Patient 1
        
        # Build the Plotly graph
        fig = go.Figure(data=go.Scatter(y=patient_wave, mode='lines', line=dict(color='#ff4b4b', width=2)))
        fig.update_layout(
            xaxis_title="Time (Milliseconds)",
            yaxis_title="Blood Volume",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0) # Makes the graph fit better on the web page
        )
        
        # Display the graph on the Streamlit website!
        st.plotly_chart(fig, use_container_width=True)
        # ----------------------
        
        st.subheader("🩺 Estimated Results:")
        st.write("**Heart Rate:** 72 BPM")
        st.write("**Stiffness Index:** 8.4")
        st.write("**Vascular Age:** 45 years")