# backend/inference/melody_extractor.py

import librosa
import soundfile as sf
from pathlib import Path
import numpy as np

from .melody_transformer import MelodyTransformer


class MelodyExtractor:
    """
    A1 模式旋律提取：
    - 对原始音频做强旋律变形（pitch + 节奏）
    - 仅保留前 3 秒作为 melody conditioning
    - 这样既有一点影子，又给 MusicGen 极大自由
    """

    def __init__(self, target_sr=32000):
        self.target_sr = target_sr
        self.transformer = MelodyTransformer(target_sr=target_sr)

    def extract_melody_to_wav(
        self,
        audio_path,
        target_style: str,
        target_emotion: str,
        strength: float = 0.9,
        output_path=None
    ):
        # 1. 加载原始音频（32kHz）
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

        # 2. 做 A1 级旋律变形（几乎认不出）
        y_trans, sr = self.transformer.transform(
            y, sr, style=target_style, emotion=target_emotion, strength=strength
        )

        # 3. 只保留前 3 秒作为 melody conditioning
        MAX_SECONDS = 3
        max_len = int(sr * MAX_SECONDS)
        if len(y_trans) > max_len:
            y_trans = y_trans[:max_len]

        # 4. 再加一点极轻噪声
        y_trans += np.random.randn(len(y_trans)) * 0.001

        # 5. 保存到 melody.wav
        if output_path is None:
            parent = Path(audio_path).parent
            output_path = parent / "melody.wav"

        sf.write(output_path, y_trans, sr)
        return str(output_path)
