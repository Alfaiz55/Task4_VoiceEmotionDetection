import os
import numpy as np
import librosa
import sounddevice as sd
import wavio
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import pickle

# --- Settings ---
SAMPLE_RATE = 22050
DURATION = 3  # seconds
FILENAME = "recorded.wav"

# --- Load Model and Label Encoder ---
model = load_model("voice_emotion_best.keras")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
print("✅ Model and LabelEncoder loaded successfully")

# --- Feature Extraction ---
def extract_features(file_path, n_mfcc=40, max_pad_len=174):
    try:
        # Load audio
        signal, sr = librosa.load(file_path, sr=None, duration=DURATION)

        # MFCC + delta + delta-delta
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # Stack → (120, time)
        mfcc_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

        # Pad/truncate
        if mfcc_features.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc_features.shape[1]
            mfcc_padded = np.pad(mfcc_features, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc_padded = mfcc_features[:, :max_pad_len]

        # Normalize
        mfcc_norm = (mfcc_padded - np.mean(mfcc_padded)) / (np.std(mfcc_padded) + 1e-9)

        # Add channel dim → (1, 120, 174, 1)
        return np.expand_dims(mfcc_norm[..., np.newaxis], axis=0)

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

# --- Prediction ---
def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        result_label.config(text="❌ Could not extract features", fg="red")
        return

    pred = model.predict(features)
    pred_class = np.argmax(pred, axis=1)[0]
    emotion_label = le.inverse_transform([pred_class])[0]
    result_label.config(text=f"🎯 Predicted Emotion: {emotion_label.upper()} voice", fg="white")

# --- Upload File ---
def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        predict_emotion(file_path)

# --- Record Audio ---
def record_audio():
    duration = DURATION
    print("🎙️ Recording...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    wavio.write(FILENAME, recording, SAMPLE_RATE, sampwidth=2)  # save file
    print("✅ Recording saved as", FILENAME)
    predict_emotion(FILENAME)

# --- GUI Setup ---
root = tk.Tk()
root.title("🎵 Voice Emotion Detection")
root.configure(bg="blue")

# Buttons
btn_predict = tk.Button(root, text="Predict", command=upload_file, bg="green", fg="white", width=20, height=2)
btn_predict.pack(pady=10)

btn_record = tk.Button(root, text="Record", command=record_audio, bg="red", fg="white", width=20, height=2)
btn_record.pack(pady=10)

btn_upload = tk.Button(root, text="Upload", command=upload_file, bg="black", fg="white", width=20, height=2)
btn_upload.pack(pady=10)

# Result label
result_label = tk.Label(root, text="Upload or Record to Predict", bg="blue", fg="white", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()
