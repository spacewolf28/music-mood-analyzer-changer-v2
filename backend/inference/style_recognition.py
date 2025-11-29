import librosa
import numpy as np
import joblib
import scipy.signal
from typing import Dict, Tuple

from backend.utils.safe_librosa import (
    safe_rms,
    safe_spectral_centroid,
    safe_chroma_stft,
    safe_spectral_contrast,
)

# 修复 librosa hann
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

MODEL_PATH = "backend/models/style_model.pkl"
ENCODER_PATH = "backend/models/style_label_encoder.pkl"

# =========================
# 全局加载模型 & encoder
# =========================
try:
    _STYLE_MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"[style_recognition] 无法加载模型：{MODEL_PATH}\n{e}"
    )

try:
    _STYLE_ENCODER = joblib.load(ENCODER_PATH)
except Exception as e:
    raise RuntimeError(
        f"[style_recognition] 无法加载标签编码器：{ENCODER_PATH}\n{e}"
    )


def extract_style_features(path: str) -> np.ndarray:
    """
    === 与训练一致的 68 维特征 ===
    """
    y, sr = librosa.load(path, sr=None, mono=True)

    # ---- tempo ----
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # ---- RMS（兼容） ----
    rms = safe_rms(y, sr).mean()

    # ---- centroid（兼容） ----
    centroid = safe_spectral_centroid(y, sr).mean()

    # ---- chroma ----
    chroma = safe_chroma_stft(y, sr)

    # ---- mel ----
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40).mean(axis=1)

    # ---- contrast ----
    contrast = safe_spectral_contrast(y, sr)

    # ---- tonnetz（部分音频会失败，兜底） ----
    try:
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(y),
            sr=sr,
        ).mean(axis=1)
    except Exception:
        tonnetz = np.zeros(6)

    feature = np.concatenate([
        [tempo, rms, centroid],
        chroma,
        mel,
        contrast,
        tonnetz,
    ])

    return feature.reshape(1, -1)


def predict_style(path: str) -> Tuple[str, Dict[str, float]]:
    feat = extract_style_features(path)

    model = _STYLE_MODEL
    encoder = _STYLE_ENCODER

    idx = model.predict(feat)[0]
    label = encoder.inverse_transform([idx])[0]

    try:
        prob = model.predict_proba(feat)[0]
        classes = encoder.classes_
        prob_dict = {classes[i]: float(prob[i]) for i in range(len(classes))}
    except Exception:
        classes = encoder.classes_
        prob_dict = {cls: (1.0 if cls == label else 0.0) for cls in classes}

    return label, prob_dict


if __name__ == "__main__":
    test_path = r"backend/test_audio.wav"
    s, p = predict_style(test_path)
    print("预测风格:", s)
    print("概率:", p)
