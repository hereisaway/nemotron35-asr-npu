"""Introspect the attention call inside the HF Nemotron encoder."""

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.models.nemotron_asr_streaming import modeling_nemotron_asr_streaming as modeling

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
    print("attn implementation:", model.config._attn_implementation)

    original = modeling.eager_attention_forward
    calls = {"n": 0}

    def traced(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            mask = attention_mask
            finite = torch.isfinite(mask)
            print(f"call {calls['n']}: q {tuple(query.shape)} k {tuple(key.shape)} "
                  f"mask {tuple(mask.shape)} dtype {mask.dtype} "
                  f"min {float(mask.min()):.2f} max {float(mask.max()):.2f} "
                  f"finite frac {float(finite.float().mean()):.3f} "
                  f"zero frac {float((mask == 0).float().mean()):.3f}")
        output = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if calls["n"] <= 2:
            attn = output[0]
            print(f"  attn out norm {float(attn.norm()):.4f} absmax {float(attn.abs().max()):.4f}")
        return output

    modeling.eager_attention_forward = traced
    try:
        audio = load_audio(CLIP)
        inputs = processor(audio, sampling_rate=16000, language="en-US")
        with torch.inference_mode():
            model(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"],
                  decoder_input_ids=torch.tensor([[13087]]), prompt_ids=inputs["prompt_ids"],
                  num_lookahead_tokens=3)
    finally:
        modeling.eager_attention_forward = original
    print("attention calls:", calls["n"])


if __name__ == "__main__":
    main()
