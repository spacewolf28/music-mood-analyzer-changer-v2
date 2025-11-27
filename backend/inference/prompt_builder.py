class PromptBuilder:
    """
    Stable prompt builder for MusicGen.
    Keep prompt short, clear, effective.
    """

    def build(self, style, emotion, melody_info):
        key = melody_info.get("key", "C")
        scale = melody_info.get("scale", "major")
        tempo = melody_info.get("tempo", 120)

        prompt = (
            f"A {style} music track with a {emotion} mood. "
            f"Key: {key} {scale}. "
            f"Tempo: around {tempo} BPM. "
            "Coherent structure, clear melody, studio-quality sound."
        )
        return prompt
