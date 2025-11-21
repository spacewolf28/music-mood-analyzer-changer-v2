# backend/inference/full_pipeline.py

from pathlib import Path

from .analyze import analyzer
from .prompt_builder import PromptBuilder
from .melody_extractor import MelodyExtractor
from .generate_music import MusicGenerator


class FullMusicPipeline:
    """
    A1 模式全流程：
    - 检测原歌 style/emotion（展示用）
    - 对旋律进行强变形（保留少量影子）
    - 构造强风格、强情绪 Prompt
    - 使用 MusicGen 生成几乎“新歌”的版本
    """

    def __init__(self):
        self.analyzer = analyzer
        self.prompt_builder = PromptBuilder()
        self.melody_extractor = MelodyExtractor()
        self.music_generator = MusicGenerator()

    def process(
        self,
        audio_path,
        target_style,
        target_emotion,
        output_dir="output",
        melody_transform_strength: float = 0.9
    ):
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 分析原歌（仅做展示，不用于约束）
        print("🔍 [1/4] Analyzing input audio...")
        analysis = self.analyzer.analyze(str(audio_path))
        print("Input Style:", analysis["style"])
        print("Input Emotion:", analysis["emotion"])

        # 2. 构建 A1 模式 Prompt
        print("\n🧠 [2/4] Building aggressive prompt...")
        prompt = self.prompt_builder.build_prompt(target_style, target_emotion)
        print(prompt)

        # 3. 提取 + 变形旋律 → 只留 3 秒
        print("\n🎼 [3/4] Extracting and transforming melody (A1 mode)...")
        melody_path = self.melody_extractor.extract_melody_to_wav(
            str(audio_path),
            target_style=target_style,
            target_emotion=target_emotion,
            strength=melody_transform_strength,
            output_path=output_dir / "melody.wav"
        )

        # 4. 生成几乎“新歌”的风格转换版本
        print("\n🎶 [4/4] Generating transformed music...")
        output_audio_path = output_dir / "generated_style_transfer.wav"

        self.music_generator.generate_with_melody(
            prompt=prompt,
            melody_path=str(melody_path),
            output_path=str(output_audio_path),
            max_new_tokens=512   # 建议 512，长度/速度比较平衡
        )

        print("\n🎉 Done! New song saved at:", output_audio_path)

        return {
            "analysis": analysis,
            "prompt": prompt,
            "output": str(output_audio_path)
        }


if __name__ == "__main__":
    print("\n===============================")
    print(" 🚀 Full Pipeline A1 (Strong Transform) ")
    print("===============================\n")

    pipeline = FullMusicPipeline()

    INPUT_AUDIO = r"D:\idea_python\music_project\backend\test_audio.wav"
    TARGET_STYLE = "pop"      # rock / jazz / classical / pop / electronic
    TARGET_EMOTION = "scary"   # angry / funny / happy / sad / scary / tender

    pipeline.process(
        audio_path=INPUT_AUDIO,
        target_style=TARGET_STYLE,
        target_emotion=TARGET_EMOTION,
        output_dir=r"D:\idea_python\music_project\backend\output",
        melody_transform_strength=0.9   # A1：0.8~0.95 建议
    )
