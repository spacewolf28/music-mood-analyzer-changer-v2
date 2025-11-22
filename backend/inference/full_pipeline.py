# backend/inference/full_pipeline.py

from pathlib import Path
import time

from .analyze import analyzer
from .prompt_builder import PromptBuilder
from .melody_extractor import MelodyExtractor
from .melody_transformer import MelodyTransformer
from .generate_music import MusicGenerator


class FullMusicPipeline:
    """
    方案 A：多 attempt 自动再生成 + 不掉音版流水线

    流程：
    - 使用你的 style/emotion 模型分析原歌
    - 使用 PromptBuilder 构造随 attempt 变化的 prompt
    - 使用 MelodyExtractor + MelodyTransformer 逐步弱化/变形旋律
    - 使用 MusicGen-medium 生成 ~15 秒音乐（带 anti-collapse）
    - 再用你的模型分析生成结果，根据 style/emotion 命中打分，选最佳版本
    """

    def __init__(self):
        self.analyzer = analyzer
        self.prompt_builder = PromptBuilder()
        self.melody_extractor = MelodyExtractor()
        self.melody_transformer = MelodyTransformer()
        self.music_generator = MusicGenerator()

    @staticmethod
    def score_generation(
        predicted_style: str,
        predicted_emotion: str,
        target_style: str,
        target_emotion: str,
    ) -> int:
        """
        简单评分：
        - style 命中 +1
        - emotion 命中 +1
        """
        score = 0
        if predicted_style and predicted_style.lower() == target_style.lower():
            score += 1
        if predicted_emotion and predicted_emotion.lower() == target_emotion.lower():
            score += 1
        return score

    @staticmethod
    def guidance_for_attempt(attempt: int) -> float:
        """
        尝试次数 → guidance_scale：
        - 1: 3.8（旋律最稳）
        - 2: 3.6
        - 3: 3.4
        - >=4: 3.2
        """
        if attempt <= 1:
            return 3.8
        elif attempt == 2:
            return 3.6
        elif attempt == 3:
            return 3.4
        else:
            return 3.2

    def process(
        self,
        audio_path: str,
        target_style: str,
        target_emotion: str,
        output_dir: str = "backend/output",
        max_attempts: int = 4,
    ):
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not audio_path.is_file():
            raise FileNotFoundError(f"Input audio not found: {audio_path}")

        # 1. 用你的模型分析原歌
        print("🔍 [1/6] Analyzing original audio with your models...")
        analysis = self.analyzer.analyze(str(audio_path))
        orig_style = analysis.get("style")
        orig_emotion = analysis.get("emotion")
        style_prob = analysis.get("style_prob")
        emotion_prob = analysis.get("emotion_prob")

        print(f"   → Input Style:   {orig_style}")
        print(f"   → Input Emotion: {orig_emotion}")

        # 全局记录最佳结果
        best_score = -1
        best_output_path: str | None = None
        best_result: dict | None = None

        print("\n🎶 [2/6] Start Auto-Regenerate loop...")
        for attempt in range(1, max_attempts + 1):
            print(f"\n================ ATTEMPT {attempt}/{max_attempts} ================")

            # 2. 构造 Prompt
            print("🧠 Building prompt...")
            prompt = PromptBuilder.build_prompt(
                target_style=target_style,
                target_emotion=target_emotion,
                orig_style=orig_style,
                orig_emotion=orig_emotion,
                style_prob=style_prob,
                emotion_prob=emotion_prob,
                attempt=attempt,
            )
            print("----- Prompt -----")
            print(prompt)
            print("------------------")

            # 3. Melody 提取（随 attempt 改变长度）+ 变形
            print("\n🎼 Extracting & transforming melody...")
            weaken_level = attempt - 1
            raw_melody_path = self.melody_extractor.extract_melody_to_wav(
                str(audio_path),
                target_style=target_style,
                target_emotion=target_emotion,
                strength=0.9,
                output_path=output_dir / f"melody_attempt_{attempt}.wav",
                weaken_level=weaken_level,
            )

            transformed_melody_path = self.melody_transformer.transform(
                raw_melody_path, attempt=attempt
            )

            # 4. 调用 MusicGen 生成
            print("\n🎧 Generating with MusicGen (medium)...")
            output_audio_path = output_dir / f"generated_attempt_{attempt}.wav"
            guidance_scale = self.guidance_for_attempt(attempt)

            self.music_generator.generate_with_melody(
                prompt=prompt,
                melody_path=str(transformed_melody_path),
                output_path=str(output_audio_path),
                target_seconds=15.0,
                guidance_scale=guidance_scale,
                temperature=1.0,
                top_p=0.95,
                do_sample=True,
            )

            # 5. 使用你的模型分析生成结果
            print("\n📊 Analyzing generated audio with your models...")
            gen_result = self.analyzer.analyze(str(output_audio_path))
            pred_style = gen_result.get("style")
            pred_emotion = gen_result.get("emotion")

            print(f"   → Generated Style:   {pred_style}")
            print(f"   → Generated Emotion: {pred_emotion}")

            score = self.score_generation(
                pred_style, pred_emotion,
                target_style, target_emotion,
            )
            print(f"➡ Score for this attempt: {score} / 2")

            # 记录最佳结果
            if score > best_score:
                best_score = score
                best_output_path = str(output_audio_path)
                best_result = gen_result

            # style + emotion 全命中可以提前 stop
            if score == 2:
                print("✨ Perfect style + emotion match! Stop early.")
                break

            time.sleep(1)

        # 6. 输出最终最佳结果
        print("\n🎉 [6/6] Final Best Result:")
        if best_result is None or best_output_path is None:
            print("⚠ 没有成功生成任何结果，请检查上面的日志。")
            return None

        print("Best Style:   ", best_result.get("style"))
        print("Best Emotion: ", best_result.get("emotion"))
        print("Best Score:   ", best_score)
        print("Best File:    ", best_output_path)

        print("\n============== CHANGE SUMMARY ==============")
        print(f"Style:   {orig_style} → {best_result.get('style')}")
        print(f"Emotion: {orig_emotion} → {best_result.get('emotion')}")
        print("===========================================\n")

        return best_output_path


if __name__ == "__main__":
    pipeline = FullMusicPipeline()

    INPUT_AUDIO = r"backend/test_audio.wav"   # 换成你的测试音频
    TARGET_STYLE = "rock"                     # rock / jazz / classical / pop / electronic
    TARGET_EMOTION = "happy"                  # angry / funny / happy / sad / scary / tender

    pipeline.process(
        audio_path=INPUT_AUDIO,
        target_style=TARGET_STYLE,
        target_emotion=TARGET_EMOTION,
        output_dir="backend/output",
        max_attempts=4,
    )
