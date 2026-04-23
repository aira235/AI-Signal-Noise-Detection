import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import add_noise, train_model, detect_and_filter

st.title("AI-Based Signal Noise Detection & Filtering System")

# Train model
if "model" not in st.session_state:
    st.session_state.model = train_model()

model = st.session_state.model

st.sidebar.header("Choose Input Type")

input_type = st.sidebar.radio("Select Input:", ["Generate Signal", "Upload Signal File"])

# =========================
# OPTION 1: GENERATE SIGNAL
# =========================
if input_type == "Generate Signal":

    freq = st.sidebar.slider("Frequency", 1, 20, 5)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.5)

    t = np.linspace(0, 1, 500)
    clean_signal = np.sin(2 * np.pi * freq * t)

    noise = np.random.normal(0, noise_level, clean_signal.shape)
    input_signal = clean_signal + noise

# =========================
# OPTION 2: UPLOAD SIGNAL
# =========================
else:
    uploaded_file = st.file_uploader("Upload CSV file with signal values", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if df.shape[1] == 1:
            input_signal = df.iloc[:, 0].values
            t = np.arange(len(input_signal))
            clean_signal = input_signal.copy()  # assumed original unknown
        else:
            st.error("CSV should contain only one column of signal values")
            st.stop()
    else:
        st.warning("Please upload a CSV file")
        st.stop()

# Detect & filter
status, output_signal = detect_and_filter(model, input_signal)

st.write(f"### Detection Result: {status}")

# Plot
fig, ax = plt.subplots()
ax.plot(t, input_signal, label="Input Signal")
ax.plot(t, output_signal, label="Filtered Signal")
ax.legend()

st.pyplot(fig)
