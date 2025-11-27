import joblib
import librosa
import numpy as np

MODEL_PATH = "backend/models/style_model.pkl"
ENCODER_PATH = "backend/models/style_label_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


def extract_style_features(path):
    y, sr = librosa.load(path, sr=None, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_mean = mel.mean(axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    feat = np.concatenate([[tempo, rms, centroid], mel_mean, chroma_mean])
    return feat.reshape(1, -1)


def predict_style(path):
    feat = extract_style_features(path)
    prob = model.predict_proba(feat)[0]
    idx = np.argmax(prob)
    label = encoder.inverse_transform([idx])[0]
    return label, prob.tolist()
