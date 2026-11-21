"""Test whether the HF port's streaming attention context is the blocker."""

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
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    clips = {
        "weather": load_audio("bcn_weather.mp3"),
        "draw": load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav"),
        "cars": load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_cars.wav"),
    }
    inputs = {name: processor(audio, sampling_rate=16000, language="en-US") for name, audio in clips.items()}
    for value in inputs.values():
        value.pop("num_lookahead_tokens", None)

    def decode(name: str, label: str, lookahead: int) -> None:
        with torch.inference_mode():
            out = model.generate(**inputs[name], num_lookahead_tokens=lookahead,
                                 return_dict_in_generate=True)
        ids = out.sequences[0].tolist()
        blanks = sum(1 for token in ids if token == BLANK_ID)
        text = processor.decode(out.sequences, skip_special_tokens=True)[0]
        print(f"{label:34s} {name:8s} tokens {len(ids):4d} blank {blanks:4d} text {text!r}")

    for lookahead in (3, 13, 57):
        decode("weather", f"lookahead {lookahead}", lookahead)
        decode("cars", f"lookahead {lookahead}", lookahead)
        decode("draw", f"lookahead {lookahead}", lookahead)

    encoder = model.encoder if hasattr(model, "encoder") else model.model.encoder
    original_window = encoder.config.sliding_window
    encoder.config.sliding_window = 5000
    print(f"sliding_window {original_window} -> {encoder.config.sliding_window} (full left context)")
    for lookahead in (3, 13, 5000):
        decode("weather", f"full-left lookahead {lookahead}", lookahead)
        decode("cars", f"full-left lookahead {lookahead}", lookahead)
        decode("draw", f"full-left lookahead {lookahead}", lookahead)
    encoder.config.sliding_window = original_window


if __name__ == "__main__":
    main()
