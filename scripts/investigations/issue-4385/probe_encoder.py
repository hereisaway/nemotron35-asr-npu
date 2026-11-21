"""Probe encoder output and attention-mask handling for Nemotron 3.5 ASR."""

import sys

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


def decode(model, processor, label: str, **kwargs) -> None:
    with torch.inference_mode():
        out = model.generate(return_dict_in_generate=True, **kwargs)
    sequences = out.sequences
    ids = sequences[0].tolist()
    blanks = sum(1 for token in ids if token == BLANK_ID)
    text = processor.decode(sequences, skip_special_tokens=True)[0]
    print(f"{label:24s} tokens {len(ids):4d} blank {blanks:4d} text {text!r}")


def main() -> None:
    path = sys.argv[1]
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    audio = load_audio(path)
    inputs = processor(audio, sampling_rate=16000, language="en-US")
    features = inputs["input_features"]
    mask = inputs["attention_mask"]
    prompt_ids = inputs["prompt_ids"]
    print("inputs:", {key: getattr(value, "shape", value) for key, value in inputs.items()}, "prompt", prompt_ids.tolist())

    with torch.inference_mode():
        encoder = model.get_audio_features(
            input_features=features,
            attention_mask=mask,
            prompt_ids=prompt_ids,
            num_lookahead_tokens=3,
            output_attention_mask=True,
        )
    print("encoder pooler", tuple(encoder.pooler_output.shape),
          "mean %.4f std %.4f" % (encoder.pooler_output.mean(), encoder.pooler_output.std()),
          "per-frame std range %.4f-%.4f" % (encoder.pooler_output.std(-1).min(), encoder.pooler_output.std(-1).max()))
    if encoder.attention_mask is not None:
        print("encoder mask sum", int(encoder.attention_mask.sum()), "shape", tuple(encoder.attention_mask.shape))

    decode(model, processor, "raw+mask+prompt", input_features=features, attention_mask=mask, prompt_ids=prompt_ids)
    decode(model, processor, "raw no mask", input_features=features, prompt_ids=prompt_ids)
    decode(model, processor, "mask None", input_features=features, attention_mask=None, prompt_ids=prompt_ids)
    decode(model, processor, "features x1e3", input_features=features * 1000.0, attention_mask=mask, prompt_ids=prompt_ids)
    decode(model, processor, "auto prompt", input_features=features, attention_mask=mask,
           prompt_ids=torch.tensor([101]))


if __name__ == "__main__":
    main()
