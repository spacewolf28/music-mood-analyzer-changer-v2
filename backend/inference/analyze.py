from .emotion_recognition import predict_emotion
from .style_recognition import predict_style
from .melody_extractor import MelodyExtractor


class Analyzer:

    def __init__(self):
        self.melody_extractor = MelodyExtractor()

    def analyze_melody(self, path):
        return self.melody_extractor.extract(path)

    def analyze(self, path):
        style, style_prob = predict_style(path)
        emotion, emotion_prob = predict_emotion(path)
        melody = self.analyze_melody(path)

        return {
            "style": style,
            "style_prob": style_prob,
            "emotion": emotion,
            "emotion_prob": emotion_prob,
            "melody": melody
        }
