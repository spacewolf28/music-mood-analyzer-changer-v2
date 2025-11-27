import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MelodyExtractor:

    def extract_pitch(self, y, sr):
        pitch, mag = librosa.piptrack(y=y, sr=sr)
        pitches = pitch[mag > np.percentile(mag, 90)]
        if len(pitches) == 0:
            return np.array([])

        return librosa.util.normalize(pitches)

    def detect_key(self, y, sr):
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_index = chroma_mean.argmax()
        keys = ["C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B"]
        return keys[key_index]

    def detect_scale(self, y, sr):
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        if tonnetz.mean() > 0:
            return "major"
        return "minor"

    def tempo(self, y, sr):
        t, _ = librosa.beat.beat_track(y=y, sr=sr)
        return int(t)

    # 防止维度不一致
    def safe_similarity(self, a, b):
        L = min(len(a), len(b))
        if L < 10:
            return 0
        return cosine_similarity([a[:L]], [b[:L]])[0][0]

    def extract(self, path):
        y, sr = librosa.load(path, sr=None, mono=True)

        pitch = self.extract_pitch(y, sr)
        key = self.detect_key(y, sr)
        scale = self.detect_scale(y, sr)
        tempo = self.tempo(y, sr)

        # phrase repetition（安全版）
        rep = 0
        if len(pitch) > 200:
            chunk = len(pitch) // 3
            p1, p2 = pitch[:chunk], pitch[chunk:2*chunk]
            rep = self.safe_similarity(p1, p2)

        return {
            "key": key,
            "scale": scale,
            "tempo": tempo,
            "phrase_repetition": float(rep)
        }
