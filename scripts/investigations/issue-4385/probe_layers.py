"""Per-layer norm trace of the Nemotron 3.5 ASR encoder."""

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


def trace(model, processor, audio: np.ndarray, name: str) -> None:
    inputs = processor(audio, sampling_rate=16000, language="en-US")
    encoder = model.encoder if hasattr(model, "encoder") else model.model.encoder
    records = []

    def describe(tensor) -> str:
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            return "n/a"
        norms = tensor.reshape(-1, tensor.shape[-1]).norm(dim=-1)
        return f"{float(norms.median()):7.2f}/{float(norms.max()):8.2f}"

    hooks = []
    for layer_index, layer in enumerate(encoder.layers):
        hooks.append(layer.register_forward_hook(
            lambda module, args, output, index=layer_index: records.append((f"layer{index}", describe(output)))))
    hooks.append(encoder.register_forward_hook(
        lambda module, args, output: records.append(("encoder", describe(output)))))

    with torch.inference_mode():
        model(
            input_features=inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=torch.tensor([[BLANK_ID]]),
            prompt_ids=inputs["prompt_ids"],
            num_lookahead_tokens=3,
        )
    for hook in hooks:
        hook.remove()
    print(f"\n{name}: median/max frame norm")
    for label, value in records:
        print(f"  {label:10s} {value}")


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    trace(model, processor, load_audio("bcn_weather.mp3"), "weather")
    trace(model, processor, load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav"), "draw")


if __name__ == "__main__":
    main()
