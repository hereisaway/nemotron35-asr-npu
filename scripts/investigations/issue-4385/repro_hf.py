"""Run the HF/vendor Nemotron 3.5 ASR path on one clip and report token statistics."""

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


def main() -> None:
    path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "en-US"
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    audio = load_audio(path)
    inputs = processor(audio, sampling_rate=16000, language=language)
    print(f"audio {path}: {len(audio) / 16000:.2f} s, features {tuple(inputs['input_features'].shape)}, "
          f"frames_valid {int(inputs['attention_mask'].sum())}, prompt_id {int(inputs['prompt_ids'][0])}")

    features = inputs["input_features"]
    print(f"features stats: mean {features.mean():.3f} std {features.std():.3f} "
          f"min {features.min():.3f} max {features.max():.3f}")

    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True)

    sequences = out.sequences
    for row in sequences:
        ids = row.tolist()
        blank = sum(1 for token in ids if token == BLANK_ID)
        pad = sum(1 for token in ids if token == 0)
        print(f"tokens {len(ids)} blank {blank} pad(0) {pad} non-blank sample {ids[:24]}")
    print("decoded:", repr(processor.decode(sequences, skip_special_tokens=True)[0]))


if __name__ == "__main__":
    main()
