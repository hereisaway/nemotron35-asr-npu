"""Diff NeMo positional encoding and conformer layer 0 against the HF port."""

import librosa
import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForRNNT, AutoProcessor

NEMO_PATH = "/opt/atomgit/models/nemo/nemotron-3.5-asr-streaming-0.6b.nemo"
MODEL_DIR = "/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b"
CLIPS = {
    "weather": "/opt/atomgit/audio/bcn_weather.mp3",
    "draw": "/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav",
}


def load_audio(path: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
    return np.asarray(audio, dtype=np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(-1, left.shape[-1])
    right = right.reshape(-1, right.shape[-1])
    size = min(left.shape[0], right.shape[0])
    left, right = left[:size], right[:size]
    return (left * right).sum(-1) / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-9)


def take(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def main() -> None:
    nemo = nemo_asr.models.ASRModel.restore_from(NEMO_PATH, map_location="cpu").eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    hf = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    hf_encoder = hf.encoder if hasattr(hf, "encoder") else hf.model.encoder

    for name, path in CLIPS.items():
        audio = load_audio(path)
        signal = torch.from_numpy(audio).unsqueeze(0)
        inputs = processor(audio, sampling_rate=16000, language="en-US")

        with torch.inference_mode():
            processed, processed_length = nemo.preprocessor(
                input_signal=signal, length=torch.tensor([signal.shape[1]]))
            nemo_sub, nemo_length = nemo.encoder.pre_encode(
                x=processed.transpose(1, 2), lengths=processed_length)
            hf_sub = hf_encoder.subsampling(inputs["input_features"], inputs["attention_mask"])

            hf_pos = hf_encoder.encode_positions(hf_sub, cached_frames=0)
            nemo_out, nemo_pos = nemo.encoder.pos_enc(x=nemo_sub, cache_len=0)

        hf_pos_np = hf_pos[0].numpy()
        nemo_pos_np = nemo_pos[0].numpy()
        pos_cos = cosine(hf_pos_np[None], nemo_pos_np[None])
        print(f"\n{name}: hf pos {hf_pos.shape} nemo pos {nemo_pos.shape} "
              f"| cosine median {np.median(pos_cos):.4f} min {pos_cos.min():.4f}")
        print(f"  norm hf {np.linalg.norm(hf_pos_np):.2f} nemo {np.linalg.norm(nemo_pos_np):.2f}")

        # Layer 0 output from a full encoder forward in each framework.
        nemo_capture = {}
        hf_capture = {}
        handle_nemo = nemo.encoder.layers[0].register_forward_hook(
            lambda module, args, output: nemo_capture.setdefault("x", take(output)))
        handle_hf = hf_encoder.layers[0].register_forward_hook(
            lambda module, args, output: hf_capture.setdefault("x", take(output)))
        with torch.inference_mode():
            nemo.encoder(audio_signal=processed, length=processed_length)
            hf(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=torch.tensor([[13087]]),
                prompt_ids=inputs["prompt_ids"],
                num_lookahead_tokens=3,
            )
        handle_nemo.remove()
        handle_hf.remove()

        nemo_layer = nemo_capture["x"]
        hf_layer = hf_capture["x"]
        nemo_layer = nemo_layer[0].numpy() if nemo_layer.ndim == 3 else nemo_layer.transpose(1, 2)[0].numpy()
        hf_layer = hf_layer[0].numpy() if hf_layer.ndim == 3 else hf_layer.transpose(1, 2)[0].numpy()
        layer_cos = cosine(hf_layer, nemo_layer)
        print(f"  layer0 hf {hf_layer.shape} nemo {nemo_layer.shape} "
              f"cosine median {np.median(layer_cos):.4f} min {layer_cos.min():.4f} "
              f"first8 {np.round(layer_cos[:8], 3).tolist()}")


if __name__ == "__main__":
    main()
