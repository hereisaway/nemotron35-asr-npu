"""Check whether the HF port's self-attention output is zero."""

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

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
    encoder = model.encoder if hasattr(model, "encoder") else model.model.encoder
    layer = encoder.layers[0]

    for name, path in CLIPS.items():
        audio = load_audio(path)
        inputs = processor(audio, sampling_rate=16000, language="en-US")
        captured = {}

        handle = layer.self_attn.register_forward_hook(
            lambda module, args, output: captured.__setitem__("out", output))
        with torch.inference_mode():
            model(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"],
                  decoder_input_ids=torch.tensor([[13087]]), prompt_ids=inputs["prompt_ids"],
                  num_lookahead_tokens=3)
        handle.remove()

        out = captured["out"]
        hook_out = out[0] if isinstance(out, (tuple, list)) else out
        print(f"{name}: output {tuple(hook_out.shape)} "
              f"norm {float(hook_out.norm()):.4f} absmax {float(hook_out.abs().max()):.4f}")
        print(f"  type {type(out).__name__} "
              f"parts {[tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__ for x in out] if isinstance(out, (tuple, list)) else '-'}")


if __name__ == "__main__":
    main()
