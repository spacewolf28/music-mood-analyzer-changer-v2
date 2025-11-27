import joblib
import librosa
import numpy as np

MODEL_PATH = "backend/models/emotion_model.pkl"
model = joblib.load(MODEL_PATH)


def extract_emotion_features(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    return mel.mean(axis=1).reshape(1, -1)


def predict_emotion(path):
    feat = extract_emotion_features(path)
    prob = model.predict_proba(feat)[0]
    idx = np.argmax(prob)
    emo = model.classes_[idx]
    return emo, prob.tolist()
