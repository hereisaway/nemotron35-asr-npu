"""Diff NeMo reference features/encoder against the HF port for #4385."""

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

        hf_features = inputs["input_features"][0].numpy()
        hf_pooled = outputs.pooler_output[0].numpy()
        nemo_features = np.load(f"/tmp/nemo_features_{name}.npy")
        nemo_pooled = np.load(f"/tmp/nemo_pooled_{name}.npy")

        frames = min(hf_features.shape[0], nemo_features.shape[0])
        feature_diff = np.abs(hf_features[:frames] - nemo_features[:frames])
        print(f"\n{name}: hf mel {hf_features.shape} nemo mel {nemo_features.shape} "
              f"max|diff| {feature_diff.max():.6f} mean|diff| {feature_diff.mean():.6f}")

        frames = min(hf_pooled.shape[0], nemo_pooled.shape[0])
        hf_norm = np.linalg.norm(hf_pooled[:frames], axis=-1)
        nemo_norm = np.linalg.norm(nemo_pooled[:frames], axis=-1)
        cosine = (hf_pooled[:frames] * nemo_pooled[:frames]).sum(-1) / (
            hf_norm * nemo_norm + 1e-9)
        print(f"  pooled hf {hf_pooled.shape} nemo {nemo_pooled.shape}")
        print(f"  frame norm hf median {np.median(hf_norm):.2f} max {hf_norm.max():.2f} | "
              f"nemo median {np.median(nemo_norm):.2f} max {nemo_norm.max():.2f}")
        print(f"  cosine median {np.median(cosine):.3f} min {cosine.min():.3f}")
        for index in range(min(8, frames)):
            print(f"    frame {index}: hf {hf_norm[index]:7.2f} nemo {nemo_norm[index]:7.2f} "
                  f"cos {cosine[index]:+.3f}")


if __name__ == "__main__":
    main()
