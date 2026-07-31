import streamlit as st

# Set up the page style
st.set_page_config(page_title="PulseForm", layout="centered")

# The Dashboard UI
st.title("🫀 PulseForm: Cardiovascular Screener")
st.write("Welcome to the PulseForm prototype. Please upload a 15-second video of your finger covering the camera and flash.")

# File uploader
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov"])

# The Fake "Integration" button
if uploaded_video is not None:
    st.success("Video uploaded successfully!")
    
    if st.button("Analyze Vascular Health"):
        with st.spinner("Extracting pulse wave... (Pretending to run Shreyas's math)"):
            import time
            time.sleep(2) # Fake loading time
            
        st.subheader("📊 Estimated Results:")
        st.write("**Heart Rate:** 72 BPM")
        st.write("**Stiffness Index:** 8.4")
        st.write("**Vascular Age:** 45 years")