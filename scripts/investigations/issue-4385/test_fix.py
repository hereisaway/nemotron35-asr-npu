"""Confirm the torch_npu mask conversion is the #4385 root cause."""

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.integrations import sdpa_attention

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

    def transcribe(clip: dict, label: str) -> None:
        with torch.inference_mode():
            out = model.generate(**clip, return_dict_in_generate=True)
        ids = out.sequences[0].tolist()
        blanks = sum(1 for token in ids if token == 13087)
        text = processor.decode(out.sequences, skip_special_tokens=True)[0]
        print(f"{label:34s} tokens {len(ids):4d} blank {blanks:4d} text {text!r}")

    inputs = {name: processor(load_audio(path), sampling_rate=16000, language="en-US")
              for name, path in CLIPS.items()}

    print(f"torch_npu detected by transformers: {sdpa_attention._is_torch_npu_available}")
    print("attn implementation:", model.config._attn_implementation)
    for name, clip in inputs.items():
        transcribe(clip, f"sdpa + npu workaround {name}")

    sdpa_attention._is_torch_npu_available = False
    print("\n-- torch_npu workaround disabled --")
    for name, clip in inputs.items():
        transcribe(clip, f"sdpa original mask {name}")

    sdpa_attention._is_torch_npu_available = True
    model.config._attn_implementation = "eager"
    model.encoder.config._attn_implementation = "eager"
    print("\n-- eager attention --")
    for name, clip in inputs.items():
        transcribe(clip, f"eager {name}")


if __name__ == "__main__":
    main()
