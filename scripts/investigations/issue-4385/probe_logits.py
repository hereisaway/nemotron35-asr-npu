"""Inspect checkpoint loading and joint-network logits for Nemotron 3.5 ASR."""

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


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model, info = AutoModelForRNNT.from_pretrained(
        MODEL_DIR, dtype=torch.float32, output_loading_info=True
    )
    model.eval()
    print("missing:", sorted(info["missing_keys"])[:10], "count", len(info["missing_keys"]))
    print("unexpected:", sorted(info["unexpected_keys"])[:10], "count", len(info["unexpected_keys"]))
    print("mismatched:", [(key, shape) for key, shape in info["mismatched_keys"]][:5])

    audio = load_audio("bcn_weather.mp3")
    inputs = processor(audio, sampling_rate=16000, language="en-US")

    with torch.inference_mode():
        outputs = model(
            input_features=inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=torch.tensor([[BLANK_ID]]),
            prompt_ids=inputs["prompt_ids"],
            num_lookahead_tokens=3,
        )
    logits = outputs.logits
    print("logits", tuple(logits.shape), "dtype", logits.dtype)
    frame_logits = logits[0, -1]
    top = torch.topk(frame_logits, 6)
    print("blank logit", float(frame_logits[BLANK_ID]))
    print("top6 ids", top.indices.tolist(), "values", [round(float(v), 3) for v in top.values])
    print("argmax over first 8 frames:")
    for step in range(8):
        step_logits = logits[0, step] if logits.shape[1] > 8 else logits[0, -1]
        best = int(step_logits.argmax())
        print(f"  frame {step}: argmax {best} blank_logit {float(step_logits[BLANK_ID]):.3f} margin "
              f"{float(step_logits[best] - step_logits[BLANK_ID]):.3f}")

    # Decoder sensitivity: joint logits for a few decoder inputs.
    with torch.inference_mode():
        for decoder_token in (BLANK_ID, 0, 101, 4000):
            out = model(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=torch.tensor([[decoder_token]]),
                prompt_ids=inputs["prompt_ids"],
                num_lookahead_tokens=3,
            )
            step = out.logits[0, -1]
            print(f"decoder {decoder_token:6d}: blank {float(step[BLANK_ID]):.3f} "
                  f"argmax {int(step.argmax())} max {float(step.max()):.3f}")


if __name__ == "__main__":
    main()
