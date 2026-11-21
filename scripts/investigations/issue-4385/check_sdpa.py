"""Inspect the SDPA mask and output inside the HF Nemotron encoder."""

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.integrations import sdpa_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

MODEL_DIR = "/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b"
CLIP = "/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav"


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
    original = sdpa_attention.sdpa_attention_forward
    calls = {"n": 0}

    def traced(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=False, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            mask = attention_mask
            finite = torch.isfinite(mask)
            print(f"call {calls['n']}: q {tuple(query.shape)} mask {tuple(mask.shape)} "
                  f"dtype {mask.dtype} min {float(mask.min()):.1f} max {float(mask.max()):.1f} "
                  f"finite {float(finite.float().mean()):.3f} is_causal {is_causal}")
            print(f"  q norm {float(query.norm()):.2f} k norm {float(key.norm()):.2f} "
                  f"v norm {float(value.norm()):.2f} finite "
                  f"{bool(torch.isfinite(query).all())}/{bool(torch.isfinite(key).all())}/{bool(torch.isfinite(value).all())}")
            row = mask[0, 0, -1]
            print(f"  last row: finite {int(torch.isfinite(row).sum())} of {row.numel()} "
                  f"zeros {int((row == 0).sum())}")
        output = original(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
        if calls["n"] <= 2:
            print(f"  out norm {float(output[0].norm()):.4f} absmax {float(output[0].abs().max()):.4f}")
        return output

    ALL_ATTENTION_FUNCTIONS["sdpa"] = traced
    try:
        audio = load_audio(CLIP)
        inputs = processor(audio, sampling_rate=16000, language="en-US")
        with torch.inference_mode():
            model(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"],
                  decoder_input_ids=torch.tensor([[13087]]), prompt_ids=inputs["prompt_ids"],
                  num_lookahead_tokens=3)
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
    print("sdpa calls:", calls["n"])


if __name__ == "__main__":
    main()
