🎤 Voice Emotion Detection
This project is part of my NullClass Internship Task 4.
It detects human emotions from voice recordings using Deep Learning (CNN + LSTM) and provides a simple GUI to test with audio files or live recording.


📂 Project Structure

├── audio.py                 # GUI for recording/uploading audio and predicting emotion  
├── voice_emotion_best.keras # Trained model file  
├── label_encoder.pkl        # Saved LabelEncoder for mapping predictions to emotion labels  
├── requirements.txt         # List of required Python libraries  
├── training_notebook.ipynb  # Jupyter notebook with full preprocessing, training & evaluation  


⚡ Features
Preprocessing using MFCC + Delta + Delta-Delta features
Balanced training with CREMA-D dataset
CNN + LSTM architecture for sequential feature learning
GUI built with Tkinter

Options in GUI:
🎙️ Record Voice (Red Button)
📂 Upload Audio File (Black Button)
✅ Predict Emotion (Green Button)

Clear output message:
Example → 🎯 Predicted Emotion: Happy voice

🎯 Emotions Detected

Angry
Happy
Sad
Fear
Neutral
Disgust

⚙️ Installation

Clone the repository and install dependencies:
git clone <your_repo_link>
cd voice_emotion_detection
pip install -r requirements.txt

📊 Model Performance::--
✅ Achieved 70%+ validation accuracy on CREMA-D dataset
Early stopping & regularization to prevent overfitting

📦 Files to Submit::--
voice_emotion_best.keras → Trained model
label_encoder.pkl → Label mapping
audio.py → GUI
requirements.txt → Dependencies
training_notebook.ipynb → Training pipeline
