"""Find the frame offset between the NeMo reference and the HF port."""

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

MODEL_DIR = "/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b"
CLIPS = {
    "weather": "/opt/atomgit/audio/bcn_weather.mp3",
    "draw": "/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav",
    "cars": "/opt/atomgit/sglang-omni/tests/data/query_to_cars.wav",
}


def load_audio(path: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
    return np.asarray(audio, dtype=np.float32)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left * right).sum(-1) / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-9)


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    for name, path in CLIPS.items():
        audio = load_audio(path)
        inputs = processor(audio, sampling_rate=16000, language="en-US")
        with torch.inference_mode():
            outputs = model(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=torch.tensor([[13087]]),
                prompt_ids=inputs["prompt_ids"],
                num_lookahead_tokens=3,
            )
        hf_pooled = outputs.pooler_output[0].numpy()
        nemo_pooled = np.load(f"/tmp/nemo_pooled_{name}.npy")
        length = min(hf_pooled.shape[0], nemo_pooled.shape[0])
        print(f"\n{name}: {length} frames")
        for shift in range(-3, 4):
            if shift >= 0:
                left = hf_pooled[: length - shift]
                right = nemo_pooled[shift:length]
            else:
                left = hf_pooled[-shift:length]
                right = nemo_pooled[: length + shift]
            cos = cosine_rows(left, right)
            print(f"  shift {shift:+d}: median {np.median(cos):.3f} mean {cos.mean():.3f} "
                  f"frac>0.95 {(cos > 0.95).mean():.2f}")

        # Per-frame best local shift
        best = []
        for index in range(length - 3):
            window = nemo_pooled[index : index + 4]
            cos = cosine_rows(np.repeat(hf_pooled[index][None], 4, axis=0), window)
            best.append(int(cos.argmax()))
        print(f"  local best shift histogram: {np.bincount(best, minlength=4).tolist()}")
        print(f"  first 24 local shifts: {best[:24]}")


if __name__ == "__main__":
    main()
