"""Diff NeMo subsampling against the HF port subsampling."""

import librosa
import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForRNNT, AutoProcessor

NEMO_PATH = "/opt/atomgit/models/nemo/nemotron-3.5-asr-streaming-0.6b.nemo"
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


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left * right).sum(-1) / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-9)


def main() -> None:
    nemo = nemo_asr.models.ASRModel.restore_from(NEMO_PATH, map_location="cpu").eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    hf = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    hf_encoder = hf.encoder if hasattr(hf, "encoder") else hf.model.encoder

    for name, path in CLIPS.items():
        audio = load_audio(path)
        signal = torch.from_numpy(audio).unsqueeze(0)
        inputs = processor(audio, sampling_rate=16000, language="en-US")

        with torch.inference_mode():
            processed, processed_length = nemo.preprocessor(
                input_signal=signal, length=torch.tensor([signal.shape[1]]))
            nemo_sub, nemo_length = nemo.encoder.pre_encode(
                x=processed.transpose(1, 2), lengths=processed_length)
            hf_sub = hf_encoder.subsampling(inputs["input_features"], inputs["attention_mask"])

        nemo_np = nemo_sub[0].numpy()
        hf_np = hf_sub[0].numpy()
        frames = min(nemo_np.shape[0], hf_np.shape[0])
        cos = cosine(hf_np[:frames], nemo_np[:frames])
        print(f"\n{name}: nemo sub {nemo_np.shape} len {int(nemo_length[0])} | hf sub {hf_np.shape}")
        print(f"  norm hf median {np.median(np.linalg.norm(hf_np, axis=-1)):.2f} "
              f"nemo median {np.median(np.linalg.norm(nemo_np, axis=-1)):.2f}")
        print(f"  cosine median {np.median(cos):.4f} min {cos.min():.4f}")
        print(f"  first 8 cos: {np.round(cos[:8], 3).tolist()}")
        direct = np.abs(hf_np[:frames] - nemo_np[:frames])
        print(f"  abs diff mean {direct.mean():.5f} max {direct.max():.4f}")


if __name__ == "__main__":
    main()
