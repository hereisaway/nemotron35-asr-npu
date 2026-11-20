"""Run one offline Nemotron 3.5 ASR transcription on Ascend NPU."""

from __future__ import annotations

import argparse
import os
import time

import librosa
import torch
import torch_npu
from sglang_omni.vendor.nemotron3_5_asr.configuration_nemotron3_5_asr import (
    Nemotron3_5AsrConfig,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)
from sglang_omni.vendor.nemotron3_5_asr.processing_nemotron3_5_asr import (
    Nemotron3_5AsrProcessor,
)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("NEMOTRON_MODEL_PATH"),
        required=os.environ.get("NEMOTRON_MODEL_PATH") is None,
    )
    parser.add_argument("--audio", default="tests/data/query_to_draw.wav")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    processor = Nemotron3_5AsrProcessor.from_pretrained(
        args.model_path, local_files_only=True
    )
    config = Nemotron3_5AsrConfig.from_pretrained(args.model_path)
    model = (
        Nemotron3_5AsrForRNNT.from_pretrained(
            args.model_path, config=config, dtype=dtype, local_files_only=True
        )
        .to("npu")
        .eval()
    )
    waveform, sample_rate = librosa.load(args.audio, sr=16000)
    inputs = processor(
        [waveform],
        sampling_rate=sample_rate,
        language=["auto"],
        padding="longest",
        return_tensors="pt",
    ).to(device="npu", dtype=dtype)

    started_at_s = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True)
    elapsed_s = time.perf_counter() - started_at_s
    sequences = output.sequences.detach().to("cpu")
    print(f"dtype={args.dtype} elapsed_s={elapsed_s:.3f} shape={tuple(sequences.shape)}")
    print(processor.batch_decode(sequences, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
