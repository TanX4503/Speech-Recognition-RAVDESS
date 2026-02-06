import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model

emotion_labels = [
    "neutral","calm","happy","sad",
    "angry","fearful","disgust","surprised"
]

def wav_to_logmel(y, sr, n_mels=128, max_len=128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)

    if logmel.shape[1] < max_len:
        logmel = np.pad(logmel, ((0,0),(0,max_len-logmel.shape[1])))
    else:
        logmel = logmel[:, :max_len]

    return logmel

model = load_model("emotion_cnn.keras")

def predict(path):
    y, sr = librosa.load(path, sr=22050)
    y, _ = librosa.effects.trim(y)

    mel = wav_to_logmel(y, sr)
    mel = mel[np.newaxis, ..., np.newaxis]

    probs = model.predict(mel)[0]
    idx = np.argmax(probs)

    print(f"Predicted Emotion: {emotion_labels[idx]}")
    print(f"Confidence: {probs[idx]*100:.2f}%")

predict("sys.argv[1]")