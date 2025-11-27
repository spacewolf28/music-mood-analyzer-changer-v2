import os
from .analyze import Analyzer
from .prompt_builder import PromptBuilder
from .generate_music import MusicGenerator


class FullPipeline:

    def __init__(self):
        self.analyzer = Analyzer()
        self.prompt_builder = PromptBuilder()
        self.generator = MusicGenerator()

    def analyze_original(self, path):
        return self.analyzer.analyze(path)

    def generate(self, style, emotion, melody, length_s):
        prompt = self.prompt_builder.build(style, emotion, melody)
        wav = self.generator.generate(prompt, length_s)
        return wav

    def run(self, input_audio, out_dir="output", length_s=10, repeat=4):
        os.makedirs(out_dir, exist_ok=True)

        print("\n=== Step 1: Analyze original ===")
        info = self.analyze_original(input_audio)
        print(info)

        style = info["style"]
        emotion = info["emotion"]
        melody = info["melody"]

        print("\n=== Step 2: Generate music ===")
        best = None
        for i in range(repeat):
            print(f"\n▶ Generating candidate {i}...")
            wav = self.generate(style, emotion, melody, length_s)
            path = f"{out_dir}/temp_out_{i}.wav"
            self.generator.save(wav, path)
            best = wav  # 简单保留最新（可改进）

        final_path = f"{out_dir}/final.wav"
        self.generator.save(best, final_path)
        print("\n🎉 All done!")
        return final_path


# ---------------------------------------
# MAIN（你要求加的）
# ---------------------------------------
def main():
    pipeline = FullPipeline()
    pipeline.run(
        input_audio="ref.wav",
        out_dir="output",
        length_s=10,
        repeat=4
    )


if __name__ == "__main__":
    main()
