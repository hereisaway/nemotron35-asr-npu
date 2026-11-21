"""Check encoder frame content alignment and leading-silence sensitivity."""

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


def encode(model, processor, audio: np.ndarray):
    inputs = processor(audio, sampling_rate=16000, language="en-US")
    with torch.inference_mode():
        outputs = model(
            input_features=inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=torch.tensor([[BLANK_ID]]),
            prompt_ids=inputs["prompt_ids"],
            num_lookahead_tokens=3,
        )
    return outputs.pooler_output[0]


def decode(model, processor, audio: np.ndarray, label: str) -> None:
    inputs = processor(audio, sampling_rate=16000, language="en-US")
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True)
    ids = out.sequences[0].tolist()
    blanks = sum(1 for token in ids if token == BLANK_ID)
    text = processor.decode(out.sequences, skip_special_tokens=True)[0]
    print(f"{label:30s} {len(audio)/16000:5.2f}s tokens {len(ids):4d} blank {blanks:4d} text {text!r}")


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    weather = load_audio("bcn_weather.mp3")
    draw = load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav")

    for name, audio in (("weather", weather), ("draw", draw)):
        frames = encode(model, processor, audio)
        print(f"{name}: frames {tuple(frames.shape)}")
        for index in range(4):
            print(f"  frame {index}: norm {float(frames[index].norm()):7.2f} head {frames[index][:4].tolist()}")

    decode(model, processor, draw, "draw base")
    decode(model, processor, np.concatenate([np.zeros(8000, np.float32), draw]), "draw +0.5s lead silence")
    decode(model, processor, np.concatenate([draw, draw]), "draw x2")
    decode(model, processor, draw[4000:], "draw -0.25s head")

    # Align test: does the encoder's frame content shift with the input length?
    frames_a = encode(model, processor, draw)
    frames_b = encode(model, processor, np.concatenate([np.zeros(800, np.float32), draw]))
    n = min(frames_a.shape[0], frames_b.shape[0]) - 1
    direct = (frames_a[:n] - frames_b[:n]).norm(dim=-1).mean()
    shifted = (frames_a[:n] - frames_b[1 : n + 1]).norm(dim=-1).mean()
    print(f"mean |diff| same-index {float(direct):.3f} shifted-by-one {float(shifted):.3f}")


if __name__ == "__main__":
    main()
