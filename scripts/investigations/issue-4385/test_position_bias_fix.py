"""Test the position_bias routing as a model-side fix for #4385."""

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

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

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def position_bias_shim(module, query, key, value, attention_mask, dropout=0.0,
                           scaling=None, is_causal=False, **kwargs):
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            return original(module, query, key, value, None, dropout, scaling,
                            is_causal, position_bias=attention_mask, **kwargs)
        return original(module, query, key, value, attention_mask, dropout, scaling,
                        is_causal, **kwargs)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = position_bias_shim
    try:
        for name, path in CLIPS.items():
            inputs = processor(load_audio(path), sampling_rate=16000, language="en-US")
            with torch.inference_mode():
                out = model.generate(**inputs, return_dict_in_generate=True)
            ids = out.sequences[0].tolist()
            blanks = sum(1 for token in ids if token == 13087)
            text = processor.decode(out.sequences, skip_special_tokens=True)[0]
            print(f"position_bias {name:8s} tokens {len(ids):4d} blank {blanks:4d} text {text!r}")
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original


if __name__ == "__main__":
    main()
