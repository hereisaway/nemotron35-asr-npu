"""Diff NeMo and HF conformer layer-0 sub-blocks."""

import librosa
import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForRNNT, AutoProcessor

NEMO_PATH = "/opt/atomgit/models/nemo/nemotron-3.5-asr-streaming-0.6b.nemo"
MODEL_DIR = "/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b"
CLIP = "/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav"
BLOCKS = ["feed_forward1", "self_attn", "conv", "feed_forward2"]


def load_audio(path: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
    return np.asarray(audio, dtype=np.float32)


def as_matrix(tensor) -> np.ndarray:
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    array = tensor.detach().numpy()
    if array.ndim == 3:
        array = array[0]
    return array


def compare(left: np.ndarray, right: np.ndarray) -> str:
    if left.shape != right.shape and left.T.shape == right.shape:
        left = left.T
    elif left.shape != right.shape and left.shape == right.T.shape:
        right = right.T
    size = min(left.shape[0], right.shape[0])
    left, right = left[:size], right[:size]
    cos = (left * right).sum(-1) / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-9)
    return f"cos median {np.median(cos):.4f} min {cos.min():.4f} | shapes {left.shape} {right.shape}"


def main() -> None:
    nemo = nemo_asr.models.ASRModel.restore_from(NEMO_PATH, map_location="cpu").eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    hf = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    hf_encoder = hf.encoder if hasattr(hf, "encoder") else hf.model.encoder

    audio = load_audio(CLIP)
    signal = torch.from_numpy(audio).unsqueeze(0)
    inputs = processor(audio, sampling_rate=16000, language="en-US")

    captured = {"nemo": {}, "hf": {}}
    handles = []
    for name in BLOCKS:
        handles.append(getattr(nemo.encoder.layers[0], name).register_forward_hook(
            lambda module, args, output, key=name: captured["nemo"].__setitem__(key, as_matrix(output))))
        handles.append(getattr(hf_encoder.layers[0], name).register_forward_hook(
            lambda module, args, output, key=name: captured["hf"].__setitem__(key, as_matrix(output))))

    with torch.inference_mode():
        processed, processed_length = nemo.preprocessor(
            input_signal=signal, length=torch.tensor([signal.shape[1]]))
        nemo.encoder(audio_signal=processed, length=processed_length)
        hf(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"],
           decoder_input_ids=torch.tensor([[13087]]), prompt_ids=inputs["prompt_ids"],
           num_lookahead_tokens=3)
    for handle in handles:
        handle.remove()

    for name in BLOCKS:
        hf_block = captured["hf"][name]
        nemo_block = captured["nemo"][name]
        print(f"{name:14s} {compare(hf_block, nemo_block)} | norms hf {np.linalg.norm(hf_block):.2f} "
              f"nemo {np.linalg.norm(nemo_block):.2f}")


if __name__ == "__main__":
    main()
