"""Test onset / DC sensitivity of the HF Nemotron 3.5 ASR path."""

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


def report(audio: np.ndarray, label: str) -> None:
    print(f"{label:28s} n {len(audio):7d} rms {np.sqrt(np.mean(audio**2)):.4f} "
          f"peak {np.abs(audio).max():.3f} dc {audio.mean():+.5f} "
          f"clip>1 {(np.abs(audio) > 1).mean() * 100:.2f}% head {audio[:5].round(3).tolist()}")


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
        print(f"{label:28s} tokens {len(ids):4d} blank {blanks:4d} text {text!r}")

    weather = load_audio("bcn_weather.mp3")
    cars = load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_cars.wav")

    report(weather, "weather raw")
    report(cars, "cars raw")

    for name, audio in (("weather", weather), ("cars", cars)):
        decode(audio, f"{name} base")
        decode(audio - audio.mean(), f"{name} dc removed")
        decode(audio[16000 // 2:], f"{name} trim 0.5s")
        decode(audio[16000:], f"{name} trim 1s")
        faded = audio.copy().astype(np.float32)
        ramp = int(0.05 * 16000)
        faded[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        decode(faded, f"{name} fade-in 50ms")
        decode(librosa.effects.preemphasis(audio), f"{name} librosa preemph")


if __name__ == "__main__":
    main()
