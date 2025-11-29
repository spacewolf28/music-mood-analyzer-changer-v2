# run_melody_test.py
# -----------------
# MelodyExtractor 测试脚本（完整可运行）
# 会：
# 1) 调用 MelodyExtractor
# 2) 输出 melody_best3s.wav
# 3) 分析输出内容（RMS + F0）
# 4) 打印是否正常

import numpy as np
import librosa
from backend.inference.melody_extractor import MelodyExtractor


def test_melody_extractor(audio_path="backend/test_audio.wav"):
    print("=== MelodyExtractor Test ===")

    extractor = MelodyExtractor()

    print(f"[1] 提取旋律片段: {audio_path}")
    out = extractor.extract_melody_to_wav(audio_path, weaken_level=0)
    print(f"[OK] 输出文件: {out}")

    print("\n[2] 加载输出音频...")
    y, sr = librosa.load(out, sr=None, mono=True)

    # ========== RMS 检查 ==========
    rms = float(np.sqrt(np.mean(y**2)))
    print(f"[分析] RMS: {rms:.6f}")

    # ========== F0 检查 ==========
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr
        )
        valid_f0 = int(np.sum(~np.isnan(f0)))
        print(f"[分析] F0 有效帧: {valid_f0}")
    except Exception as e:
        print("[错误] F0 提取失败:", e)
        valid_f0 = 0

    # ========== 判断结果 ==========
    print("\n=== 结果判断 ===")

    if rms < 0.0005:
        print("❌ 静音：RMS 极低")
    elif rms < 0.01:
        print("⚠️ 声音过小（可能是合成问题）")
    else:
        print("✔ 音量正常")

    if valid_f0 < 5:
        print("⚠️ F0 无效：旋律信息弱，正在 fallback 到原音")
    else:
        print("✔ F0 正常：存在可识别的音高结构")

    if rms > 0.01 and valid_f0 >= 5:
        print("\n🎉 结果：MelodyExtractor 输出正常！可以用于 MusicGen")
    else:
        print("\n⚠️ 检测到异常：请上传输出 WAV，我帮你进一步分析")

    print("\n=== 测试结束 ===")


if __name__ == "__main__":
    test_melody_extractor()
