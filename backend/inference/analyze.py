# backend/inference/analyze.py

from pathlib import Path
from .emotion_recognition import predict_emotion
from .style_recognition import predict_style


class Analyzer:
    def __init__(self):
        self.root = Path(__file__).resolve().parent.parent

    def analyze(self, audio_path: str) -> dict:
        audio_path = str(audio_path)

        # 风格、概率
        style, style_prob = predict_style(audio_path)

        # 情绪、概率
        emotion, emotion_prob = predict_emotion(audio_path)

        return {
            "style": style,
            "emotion": emotion,
            "style_prob": style_prob,
            "emotion_prob": emotion_prob
        }


# 全局单例
analyzer = Analyzer()
