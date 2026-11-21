"""Length sweep for the HF Nemotron 3.5 ASR path."""

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


def run(model, processor, audio: np.ndarray, label: str) -> None:
    inputs = processor(audio, sampling_rate=16000, language="en-US")
    features = inputs["input_features"]
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True)
    sequences = out.sequences
    ids = sequences[0].tolist()
    blanks = sum(1 for token in ids if token == BLANK_ID)
    text = processor.decode(sequences, skip_special_tokens=True)[0]
    valid = features[inputs["attention_mask"].bool()]
    print(f"{label:28s} {len(audio)/16000:5.2f}s frames {valid.shape[0]:4d} "
          f"mel mean {valid.mean():7.2f} tokens {len(ids):4d} blank {blanks:4d} text {text!r}")


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    weather = load_audio("bcn_weather.mp3")
    draw = load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav")

    for seconds in (3, 5, 7, 9, 11):
        run(model, processor, weather[: 16000 * seconds], f"weather {seconds}s")

    for seconds in (5, 8, 11):
        padded = np.concatenate([draw, np.zeros(max(0, 16000 * seconds - len(draw)), dtype=np.float32)])
        run(model, processor, padded, f"draw padded {seconds}s")

    gain = weather / (np.abs(weather).max() + 1e-9) * 0.9
    run(model, processor, gain, "weather peak-normalized")

    run(model, processor, weather * 0.25, "weather gain x0.25")
    run(model, processor, weather * 4.0, "weather gain x4")


if __name__ == "__main__":
    main()
