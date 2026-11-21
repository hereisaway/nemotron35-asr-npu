"""NeMo reference transcripts and encoder outputs for the #4385 clips."""

import librosa
import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr

NEMO_PATH = "/opt/atomgit/models/nemo/nemotron-3.5-asr-streaming-0.6b.nemo"
CLIPS = {
    "weather": "/opt/atomgit/audio/bcn_weather.mp3",
    "draw": "/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav",
    "cars": "/opt/atomgit/sglang-omni/tests/data/query_to_cars.wav",
}
PROMPT_INDEX = 0  # en-US


def load_audio(path: str) -> torch.Tensor:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)


def main() -> None:
    model = nemo_asr.models.ASRModel.restore_from(NEMO_PATH, map_location="cpu")
    model.eval()
    print("model:", type(model).__name__)
    print("att_context_size:", model.encoder.att_context_size,
          "| concat prompts:", getattr(model, "concat", None),
          "| num_prompts:", getattr(model, "num_prompts", None))

    for name, path in CLIPS.items():
        signal = load_audio(path)
        length = torch.tensor([signal.shape[1]])
        with torch.inference_mode():
            processed, processed_length = model.preprocessor(input_signal=signal, length=length)
            encoded, encoded_length = model.forward(
                processed_signal=processed,
                processed_signal_length=processed_length,
                prompt_indices=torch.tensor([PROMPT_INDEX]),
            )
            hypothesis = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_length,
                return_hypotheses=True,
            )
            pooled = model.joint.enc(encoded.transpose(1, 2))
        item = hypothesis[0] if isinstance(hypothesis, tuple) else hypothesis[0]
        text = item.text if hasattr(item, "text") else str(item)
        tokens = item.y_sequence.tolist() if hasattr(item, "y_sequence") else []
        np.save(f"/tmp/nemo_encoder_{name}.npy", encoded[0].transpose(0, 1).numpy())
        np.save(f"/tmp/nemo_pooled_{name}.npy", pooled[0].numpy())
        np.save(f"/tmp/nemo_features_{name}.npy", processed[0].transpose(0, 1).numpy())
        print(f"{name:8s} audio {signal.shape[1]/16000:5.2f}s mel {tuple(processed.shape)} "
              f"encoder {tuple(encoded.shape)} tokens {len(tokens)} text {text!r}")


if __name__ == "__main__":
    main()
