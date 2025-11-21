# backend/inference/evaluate_generated.py

from pathlib import Path
from backend.inference.analyze import analyzer   # 使用你的 Analyzer 单例

# ============================
# 用户可修改路径
# ============================
ORIGINAL_AUDIO = r"D:\idea_python\music_project\backend\test_audio.wav"
GENERATED_AUDIO = r"D:\idea_python\music_project\backend\output\generated_style_transfer.wav"
# ============================


def compare_style_emotion(orig, gen):
    lines = []

    # Style diff
    if orig["style"] != gen["style"]:
        lines.append(f"🎸 Style changed: {orig['style']} → {gen['style']}")
    else:
        lines.append(f"🎸 Style unchanged: {orig['style']}")

    # Emotion diff
    if orig["emotion"] != gen["emotion"]:
        lines.append(f"🎭 Emotion changed: {orig['emotion']} → {gen['emotion']}")
    else:
        lines.append(f"🎭 Emotion unchanged: {orig['emotion']}")

    return "\n".join(lines)


def main():
    print("\n==============================")
    print("      Evaluate Generated Audio")
    print("==============================\n")

    # ========= 原歌 ==========
    print("🎼 Analyzing ORIGINAL audio...\n")
    orig = analyzer.analyze(ORIGINAL_AUDIO)

    print(f"Original Style:   {orig['style']}")
    print(f"Original Emotion: {orig['emotion']}")

    # ========= 生成歌 ==========
    print("\n🎶 Analyzing GENERATED audio...\n")

    if not Path(GENERATED_AUDIO).exists():
        print(f"❌ ERROR: File not found:\n{GENERATED_AUDIO}")
        return

    gen = analyzer.analyze(GENERATED_AUDIO)

    print(f"Generated Style:   {gen['style']}")
    print(f"Generated Emotion: {gen['emotion']}")

    # ========= 对比 ==========
    print("\n==============================")
    print("           DIFFERENCE")
    print("==============================\n")
    print(compare_style_emotion(orig, gen))

    print("\n🎯 Evaluation complete.\n")


if __name__ == "__main__":
    main()
