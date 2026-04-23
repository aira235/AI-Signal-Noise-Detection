import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import add_noise, train_model, detect_and_filter

st.title("AI-Based Signal Noise Detection & Filtering System")

# Train model
if "model" not in st.session_state:
    st.session_state.model = train_model()

model = st.session_state.model

# USER INPUTS 🔥
st.sidebar.header("Input Signal Parameters")

freq = st.sidebar.slider("Frequency", 1, 20, 5)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.5)

# Generate signal based on user input
t = np.linspace(0, 1, 500)
clean_signal = np.sin(2 * np.pi * freq * t)

# Add noise based on user control
noise = np.random.normal(0, noise_level, clean_signal.shape)
input_signal = clean_signal + noise

# Detect & filter
status, output_signal = detect_and_filter(model, input_signal)

st.write(f"### Detection Result: {status}")

# Plot
fig, ax = plt.subplots()
ax.plot(t, clean_signal, label="Original Signal")
ax.plot(t, input_signal, label="Input Signal")
ax.plot(t, output_signal, label="Filtered Signal")
ax.legend()

st.pyplot(fig)