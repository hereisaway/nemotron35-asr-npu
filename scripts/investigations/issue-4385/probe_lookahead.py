"""Lookahead sweep for the HF Nemotron 3.5 ASR path."""

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

    for lookahead in (0, 3, 6, 13):
        processor.set_num_lookahead_tokens(lookahead)
        for name, audio in clips.items():
            inputs = processor(audio, sampling_rate=16000, language="en-US")
            with torch.inference_mode():
                out = model.generate(**inputs, return_dict_in_generate=True)
            ids = out.sequences[0].tolist()
            blanks = sum(1 for token in ids if token == BLANK_ID)
            text = processor.decode(out.sequences, skip_special_tokens=True)[0]
            print(f"lookahead {lookahead:2d} {name:8s} tokens {len(ids):4d} blank {blanks:4d} text {text!r}")


if __name__ == "__main__":
    main()
