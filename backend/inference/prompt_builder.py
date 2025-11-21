# backend/inference/prompt_builder.py

class PromptBuilder:
    """
    A1 模式 Prompt：
    - 强制要求重写和声、节奏、乐器编配
    - 只保留被变形后的 melody 轮廓
    """

    STYLE_INFO = {
        "rock": {
            "instrument": "distorted electric guitars, aggressive drums, heavy bass",
            "harmony": "dense rock power-chords and riffs",
            "feel": "energetic, raw, punchy"
        },
        "jazz": {
            "instrument": "saxophone, upright bass, jazz piano, brushed drums",
            "harmony": "extended jazz chords, colorful harmony and swing rhythm",
            "feel": "smooth, expressive, sophisticated"
        },
        "classical": {
            "instrument": "orchestral strings, brass, woodwinds and grand piano",
            "harmony": "rich classical orchestral harmony",
            "feel": "cinematic, dramatic, elegant"
        },
        "pop": {
            "instrument": "bright synths, tight drums, electronic bass and modern FX",
            "harmony": "catchy pop chord progressions",
            "feel": "clean, modern, radio-ready"
        },
        "electronic": {
            "instrument": "synth leads, EDM drums, sub bass and sound design",
            "harmony": "futuristic and driving harmonic motion",
            "feel": "energetic, synthetic, powerful"
        },
    }

    EMOTION_INFO = {
        "angry":    "aggressive, intense, dark emotions",
        "funny":    "playful, quirky, humorous mood",
        "happy":    "bright, uplifting, joyful energy",
        "sad":      "melancholic, emotional, minor-key feeling",
        "scary":    "tense, suspenseful, unsettling atmosphere",
        "tender":   "warm, gentle, intimate and delicate tone"
    }

    @classmethod
    def build_prompt(cls, style, emotion):
        style = style.lower()
        emotion = emotion.lower()

        s = cls.STYLE_INFO[style]
        e = cls.EMOTION_INFO[emotion]

        prompt = (
            f"Create a heavily transformed {style} style reinterpretation expressing {e}. "
            f"Use {s['instrument']} and {s['harmony']}. "
            f"Completely rewrite the original harmony, chord progressions, bassline, "
            f"drum patterns and overall arrangement in a strong {style} style. "
            f"Do NOT reuse the original arrangement, instrumentation, mix or timbre. "
            f"Only keep a faint trace of the melodic contour from the provided melody, "
            f"which has already been transformed. "
            f"Make the track feel like a new {style} piece with {s['feel']} character, "
            f"with bold stylistic deviation and clearly different mood from the original song."
        )

        return prompt
