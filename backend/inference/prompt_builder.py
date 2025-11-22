# backend/inference/prompt_builder.py

class PromptBuilder:
    """
    方案 A：多 attempt 用的最终 Prompt 构造器

    - 使用你的模型分析得到：
        * 原始风格 orig_style
        * 原始情绪 orig_emotion
    - 使用用户指定目标：
        * target_style
        * target_emotion
    - 随 attempt 变化：
        * 旋律保留比例（从保守到弱化）
        * 风格/情绪强化程度
    - 加入防静音 / 连续性 / 高质量混音描述
    """

    STYLE_PROFILE = {
        "pop": {
            "tempo": "medium tempo (~110 BPM)",
            "energy": "balanced dynamics",
            "brightness": "clean and modern tone",
            "harmony": "catchy pop chord progressions",
            "texture": "bright synth and pad textures",
            "instrument": "modern drums, bright synths, electric bass, piano, pads",
        },
        "rock": {
            "tempo": "fast tempo (~130–150 BPM)",
            "energy": "strong, punchy dynamics",
            "brightness": "bright guitar-driven tone",
            "harmony": "power chords and energetic rock riffs",
            "texture": "high-frequency guitar texture",
            "instrument": "distorted electric guitars, aggressive drums, bass",
        },
        "jazz": {
            "tempo": "slow to medium tempo (~80–110 BPM)",
            "energy": "smooth, laid-back dynamics",
            "brightness": "warm, mellow tone",
            "harmony": "extended jazz chords and ii–V–I progressions",
            "texture": "rich mid-frequency harmonic texture",
            "instrument": "upright bass, jazz piano, brushed drums, saxophone",
        },
        "classical": {
            "tempo": "expressive, flexible tempo",
            "energy": "wide dynamic range",
            "brightness": "balanced orchestral tone",
            "harmony": "classical orchestration and voice leading",
            "texture": "full-spectrum orchestral texture",
            "instrument": "strings, woodwinds, brass, piano",
        },
        "electronic": {
            "tempo": "steady fast tempo (~120–140 BPM)",
            "energy": "high intensity with strong low end",
            "brightness": "bright synthetic tone",
            "harmony": "simple, repetitive EDM-style harmony",
            "texture": "sidechain pumping synth textures",
            "instrument": "synth leads, pads, EDM drums, sub bass, FX",
        },
    }

    EMOTION_PROFILE = {
        "happy": "bright, uplifting, joyful feeling",
        "sad": "soft, emotional, melancholic mood",
        "angry": "intense, aggressive, high-energy emotion",
        "scary": "dark, tense, suspenseful atmosphere",
        "tender": "warm, intimate, gentle emotion",
        "funny": "playful, quirky, humorous character",
    }

    @classmethod
    def _describe_style(cls, style: str) -> str:
        style = (style or "").lower()
        s = cls.STYLE_PROFILE.get(style)
        if not s:
            return f"{style} style music"
        return (
            f"{s['tempo']}, {s['energy']}, {s['brightness']}, "
            f"{s['harmony']}, {s['texture']}. Typical instruments: {s['instrument']}."
        )

    @classmethod
    def build_prompt(
        cls,
        target_style: str,
        target_emotion: str,
        orig_style: str | None = None,
        orig_emotion: str | None = None,
        style_prob: dict | None = None,
        emotion_prob: dict | None = None,
        attempt: int = 1,
    ) -> str:
        target_style = (target_style or "").lower().strip()
        target_emotion = (target_emotion or "").lower().strip()

        style_desc = cls._describe_style(target_style)
        emotion_desc = cls.EMOTION_PROFILE.get(target_emotion, target_emotion)

        # 1) 原歌曲描述（来自你的模型）
        header_parts = []
        if orig_style or orig_emotion:
            parts = []
            if orig_style:
                parts.append(f"{orig_style} style")
            if orig_emotion:
                parts.append(f"{orig_emotion} emotion")
            header_parts.append(
                "The input audio is automatically analyzed as having "
                f"{' and '.join(parts)}. "
            )

        if style_prob:
            header_parts.append(f"Style distribution: {style_prob}. ")
        if emotion_prob:
            header_parts.append(f"Emotion distribution: {emotion_prob}. ")

        header_parts.append(
            f"Transform it into {target_style} style music expressing {emotion_desc}. "
        )
        header = "".join(header_parts)

        # 2) 风格模板
        style_part = (
            f"In the target {target_style} style, use the following characteristics: "
            f"{style_desc} "
        )

        # 3) 旋律保留比例随 attempt 变化
        if attempt <= 1:
            melody_part = (
                "Preserve around 60–70% of the original melodic contour so the track is clearly "
                "recognizable, but re-orchestrate it in the target style. "
            )
        elif attempt == 2:
            melody_part = (
                "Preserve around 40–60% of the original melodic contour, allowing noticeable "
                "variation in rhythm and phrasing while staying recognizable. "
            )
        elif attempt == 3:
            melody_part = (
                "Preserve only around 20–40% of the original melodic contour, focusing more on "
                "the target style and emotion while keeping a subtle hint of the original idea. "
            )
        else:
            melody_part = (
                "Preserve only a small trace of the original melodic contour; prioritize the "
                "target style and emotion, letting the music evolve more freely. "
            )

        # 4) 风格/情绪强化
        if attempt <= 1:
            strength = (
                f"Clearly emphasize {target_style} style and {target_emotion} emotion, while "
                "keeping the musical flow smooth and coherent. "
            )
        elif attempt == 2:
            strength = (
                f"Make the {target_style} style and {target_emotion} emotion very obvious to the "
                "listener, even more than in the original track. "
            )
        elif attempt == 3:
            strength = (
                f"Strongly prioritize {target_style} aesthetics and {target_emotion} expression, "
                "even if it changes the original character. "
            )
        else:
            strength = (
                f"Aggressively push towards pure {target_style} aesthetics with unmistakable "
                f"{target_emotion} emotion, letting the music feel like a strong stylistic remake. "
            )

        # 5) 防静音 + 连续性 + 质量
        quality = (
            "Ensure continuous musical texture throughout the entire duration, with no silent or "
            "empty sections and no sudden dropouts of energy. High-quality studio mix, balanced EQ, "
            "wide stereo image, expressive dynamics, instrumental only, no vocals, no noise or clipping. "
        )

        return header + style_part + melody_part + strength + quality
