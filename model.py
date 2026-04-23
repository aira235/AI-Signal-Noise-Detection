import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier

def generate_signal(freq=5, size=500):
    t = np.linspace(0, 1, size)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

def add_noise(signal):
    noise = np.random.normal(0, 0.5, signal.shape)
    return signal + noise

def extract_features(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    energy = np.sum(signal**2)
    return [mean, variance, energy]

def train_model():
    X, y = [], []
    for _ in range(200):
        t, clean = generate_signal()
        noisy = add_noise(clean)

        X.append(extract_features(clean))
        y.append(0)

        X.append(extract_features(noisy))
        y.append(1)

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def detect_and_filter(model, signal):
    prediction = model.predict([extract_features(signal)])

    if prediction[0] == 1:
        filtered = savgol_filter(signal, 51, 3)
        return "Noisy", filtered
    else:
        return "Clean", signal
