"""Energy-based trimming test for the HF Nemotron 3.5 ASR path."""

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


def leading_silence(audio: np.ndarray, top_db: float = 30.0) -> float:
    _, (start, end) = librosa.effects.trim(audio, top_db=top_db)
    return start / 16000, (len(audio) - end) / 16000


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    def decode(audio: np.ndarray, label: str) -> None:
        inputs = processor(audio, sampling_rate=16000, language="en-US")
        with torch.inference_mode():
            out = model.generate(**inputs, return_dict_in_generate=True)
        ids = out.sequences[0].tolist()
        blanks = sum(1 for token in ids if token == BLANK_ID)
        text = processor.decode(out.sequences, skip_special_tokens=True)[0]
        print(f"{label:32s} {len(audio)/16000:5.2f}s tokens {len(ids):4d} blank {blanks:4d} text {text!r}")

    clips = {
        "weather": load_audio("bcn_weather.mp3"),
        "draw": load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav"),
        "cars": load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_cars.wav"),
    }

    for name, audio in clips.items():
        head, tail = leading_silence(audio)
        print(f"{name}: lead silence {head:.2f}s tail silence {tail:.2f}s")
        decode(audio, f"{name} base")
        trimmed, _ = librosa.effects.trim(audio, top_db=30)
        decode(trimmed, f"{name} trim 30dB")
        trimmed40, _ = librosa.effects.trim(audio, top_db=40)
        decode(trimmed40, f"{name} trim 40dB")
        decode(np.concatenate([np.zeros(4000, np.float32), trimmed]), f"{name} trim +0.25s lead")


if __name__ == "__main__":
    main()
