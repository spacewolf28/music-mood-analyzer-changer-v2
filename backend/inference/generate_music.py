import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.signal import butter, filtfilt


class MusicGenerator:
    def __init__(self, model_name="facebook/musicgen-small", device="cuda"):
        print("🔧 Loading MusicGen model...")
        self.device = device if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager"
        ).to(self.device)

    # -------------------------------
    # HIGH-PASS FILTER (STABLE)
    # -------------------------------
    def highpass(self, wav, sr=32000, cutoff=80):
        b, a = butter(4, cutoff / (sr / 2), btype="high")
        return filtfilt(b, a, wav)

    # -------------------------------
    # POST PROCESSING
    # -------------------------------
    def post_process(self, wav):
        wav = wav - np.mean(wav)
        wav = self.highpass(wav, cutoff=80)
        wav = wav / (np.max(np.abs(wav)) + 1e-9)
        return wav

    # -------------------------------
    # GENERATE ONCE
    # -------------------------------
    def generate_once(self, prompt, length_s):
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        audio_values = self.model.generate(
            **inputs,
            max_new_tokens=length_s * 50,
            do_sample=True,
            temperature=1.0,
            guidance_scale=3.0
        )

        return audio_values[0, 0].cpu().numpy()

    # -------------------------------
    # MAIN GENERATION ENTRY
    # -------------------------------
    def generate(self, prompt, length_s):
        print("\n🎵 === Generating Music ===")
        print("Prompt:", prompt)

        wav = self.generate_once(prompt, length_s)

        # auto fallback if too short
        if len(wav) < (length_s * 32000 * 0.7):
            print("⚠ 输出过短，切换安全 prompt 重生成...")
            safe_prompt = (
                f"A coherent music track. Style: {prompt}. "
                f"Full length, stable rhythm."
            )
            wav = self.generate_once(safe_prompt, length_s)

        # pad to full length
        if len(wav) < length_s * 32000:
            shortage = length_s * 32000 - len(wav)
            wav = np.concatenate([wav, np.zeros(int(shortage))])

        wav = self.post_process(wav)
        return wav

    # -------------------------------
    # SAVE
    # -------------------------------
    def save(self, wav, path="output.wav"):
        sf.write(path, wav, 32000)
        print("💾 Saved:", path)
