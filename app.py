import streamlit as st
import matplotlib.pyplot as plt
from model import generate_signal, add_noise, train_model, detect_and_filter

st.title("AI-Based Signal Noise Detection & Filtering System")

if "model" not in st.session_state:
    st.session_state.model = train_model()

model = st.session_state.model

t, signal = generate_signal()

option = st.selectbox("Select Input Type:", ["Clean Signal", "Noisy Signal"])

if option == "Noisy Signal":
    input_signal = add_noise(signal)
else:
    input_signal = signal

status, output_signal = detect_and_filter(model, input_signal)

st.write(f"### Detection Result: {status}")

fig, ax = plt.subplots()
ax.plot(t, signal, label="Original")
ax.plot(t, input_signal, label="Input")
ax.plot(t, output_signal, label="Output")
ax.legend()

st.pyplot(fig)
