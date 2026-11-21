"""Compare HF mel features with a NeMo-style reference and decode from both."""

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

MODEL_DIR = "/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b"
BLANK_ID = 13087
N_FFT = 512
HOP = 160
WIN = 400
MELS = 128
SAMPLE_RATE = 16000
LOG_ZERO_GUARD = 2**-24


def load_audio(path: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def nemo_features(audio: np.ndarray, normalize: bool) -> torch.Tensor:
    waveform = torch.from_numpy(audio).float()
    preemphasized = torch.cat([waveform[:1], waveform[1:] - 0.97 * waveform[:-1]])
    window = torch.hann_window(WIN, periodic=False)
    stft = torch.stft(preemphasized, N_FFT, hop_length=HOP, win_length=WIN, window=window,
                      return_complex=True, center=True, pad_mode="constant")
    power = torch.view_as_real(stft).pow(2).sum(-1)
    mel_filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=MELS,
                                      fmin=0.0, fmax=SAMPLE_RATE / 2, norm="slaney")
    mel = torch.from_numpy(mel_filters).float() @ power
    log_mel = torch.log(mel + LOG_ZERO_GUARD).transpose(0, 1)
    if normalize:
        mean = log_mel.mean(dim=0, keepdim=True)
        var = log_mel.var(dim=0, unbiased=False, keepdim=True)
        log_mel = (log_mel - mean) / torch.sqrt(var + 1e-5)
    return log_mel.unsqueeze(0)


def decode(model, processor, features: torch.Tensor, label: str) -> None:
    mask = torch.ones(features.shape[:2], dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(input_features=features, attention_mask=mask,
                             prompt_ids=torch.tensor([0]), num_lookahead_tokens=3,
                             return_dict_in_generate=True)
    ids = out.sequences[0].tolist()
    blanks = sum(1 for token in ids if token == BLANK_ID)
    text = processor.decode(out.sequences, skip_special_tokens=True)[0]
    print(f"{label:26s} tokens {len(ids):4d} blank {blanks:4d} text {text!r}")


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    clips = {
        "weather": load_audio("bcn_weather.mp3"),
        "draw": load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_draw.wav"),
        "cars": load_audio("/opt/atomgit/sglang-omni/tests/data/query_to_cars.wav"),
    }

    for name, audio in clips.items():
        hf_inputs = processor(audio, sampling_rate=SAMPLE_RATE, language="en-US")
        hf_features = hf_inputs["input_features"]
        hf_valid = hf_features[hf_inputs["attention_mask"].bool()]

        reference = nemo_features(audio, normalize=False)
        reference_norm = nemo_features(audio, normalize=True)
        frames = min(hf_valid.shape[0], reference.shape[1])
        raw_diff = (hf_valid[:frames] - reference[0, :frames]).abs()
        print(f"{name}: hf frames {hf_valid.shape[0]} nemo frames {reference.shape[1]} "
              f"max|diff| unnormalized {float(raw_diff.max()):.4f} mean {float(raw_diff.mean()):.4f}")
        hf_stats = hf_valid.mean(0)
        nemo_stats = reference[0].mean(0)
        print(f"   per-bin mean range HF [{float(hf_stats.min()):.2f}, {float(hf_stats.max()):.2f}] "
              f"NeMo [{float(nemo_stats.min()):.2f}, {float(nemo_stats.max()):.2f}]")

        decode(model, processor, hf_features, f"{name} hf raw")
        decode(model, processor, reference_norm, f"{name} nemo normalized")


if __name__ == "__main__":
    main()
