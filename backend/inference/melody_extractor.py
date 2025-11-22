# backend/inference/melody_extractor.py

import librosa
import soundfile as sf
from pathlib import Path
import numpy as np


class MelodyExtractor:
    """
    多 attempt 强旋律提取器：

    - attempt 1: 截取前 6 秒
    - attempt 2: 截取前 5 秒
    - attempt 3: 截取前 4 秒
    - attempt >=4: 截取前 3 秒（不能再短，否则极易掉音）

    不做 pitch shift / time stretch / 噪声，保持旋律清晰稳定，
    这些变化交给 MelodyTransformer 在后面做。
    """

    def __init__(self, target_sr: int = 32000):
        self.target_sr = target_sr

    def _seconds_for_attempt(self, attempt: int) -> float:
        if attempt <= 1:
            return 6.0
        elif attempt == 2:
            return 5.0
        elif attempt == 3:
            return 4.0
        else:
            return 3.0

    def extract_melody_to_wav(
        self,
        audio_path: str,
        target_style: str | None = None,
        target_emotion: str | None = None,
        strength: float = 0.9,
        output_path: str | Path | None = None,
        weaken_level: int = 0,
    ) -> str:
        """
        参数保持兼容旧接口：
        - weaken_level: 用作 attempt-1 的索引
        """
        audio_path = str(audio_path)
        attempt = weaken_level + 1
        mel_seconds = self._seconds_for_attempt(attempt)

        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

        # 截取前 mel_seconds 秒
        max_len = int(sr * mel_seconds)
        if len(y) > max_len:
            y = y[:max_len]

        # 归一化到峰值 0.9
        max_abs = float(np.max(np.abs(y)))
        if max_abs > 1e-6:
            y = y / max_abs * 0.9

        # 输出路径
        if output_path is None:
            parent = Path(audio_path).parent
            output_path = parent / f"melody_attempt_{attempt}.wav"
        else:
            output_path = Path(output_path)

        sf.write(str(output_path), y, sr)
        print(
            f"[MelodyExtractor] Saved melody for attempt {attempt} to {output_path} "
            f"({mel_seconds:.1f}s at {sr} Hz)"
        )

        return str(output_path)
