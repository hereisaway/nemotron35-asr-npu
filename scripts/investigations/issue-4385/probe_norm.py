"""Compare raw vs normalized mel features on the HF Nemotron 3.5 ASR path."""

import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

MODEL_DIR = "/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b"
BLANK_ID = 13087


def load_audio(path: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
    return np.asarray(audio, dtype=np.float32)


def run(model, processor, features: torch.Tensor, attention_mask: torch.Tensor, label: str) -> None:
    with torch.inference_mode():
        out = model.generate(input_features=features, attention_mask=attention_mask,
                             return_dict_in_generate=True)
    sequences = out.sequences
    ids = sequences[0].tolist()
    blanks = sum(1 for token in ids if token == BLANK_ID)
    decoded = processor.decode(sequences, skip_special_tokens=True)[0]
    print(f"{label:16s} tokens {len(ids):4d} blank {blanks:4d} text {decoded!r}")


def main() -> None:
    path = sys.argv[1]
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    audio = load_audio(path)
    inputs = processor(audio, sampling_rate=16000, language="en-US")
    features = inputs["input_features"].clone()
    mask = inputs["attention_mask"].bool()
    valid = features[mask]  # (valid_frames, 128)

    print(f"audio {path}: {len(audio) / 16000:.2f} s, valid frames {valid.shape[0]}")
    print(f"valid stats: mean {valid.mean():.3f} std {valid.std():.3f} "
          f"min {valid.min():.3f} max {valid.max():.3f}")
    per_bin_mean = valid.mean(0)
    per_bin_std = valid.std(0)
    print(f"per-bin mean range [{per_bin_mean.min():.2f}, {per_bin_mean.max():.2f}] "
          f"std range [{per_bin_std.min():.2f}, {per_bin_std.max():.2f}]")

    run(model, processor, features, inputs["attention_mask"], "raw")

    per_feature = (features - per_bin_mean) / (per_bin_std + 1e-5)
    per_feature = per_feature * mask.unsqueeze(-1)
    run(model, processor, per_feature, inputs["attention_mask"], "per_feature")

    global_norm = (features - valid.mean()) / (valid.std() + 1e-5)
    global_norm = global_norm * mask.unsqueeze(-1)
    run(model, processor, global_norm, inputs["attention_mask"], "global")

    # NeMo's per_feature normalization uses the variance including a small constant
    per_feature_var = (features - per_bin_mean) / torch.sqrt(per_bin_std.pow(2) + 1e-5)
    per_feature_var = per_feature_var * mask.unsqueeze(-1)
    run(model, processor, per_feature_var, inputs["attention_mask"], "per_feature_var")


if __name__ == "__main__":
    main()
