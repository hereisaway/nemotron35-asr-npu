"""Run the eager Nemotron 3.5 ASR streaming path on Ascend NPU."""

from __future__ import annotations

import argparse
import os
import time

import librosa
import numpy as np
from sglang_omni.models.nemotron3_5_asr.model_runner import Nemotron3_5ASRModelRunner

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("NEMOTRON_MODEL_PATH"),
        required=os.environ.get("NEMOTRON_MODEL_PATH") is None,
    )
    parser.add_argument("--audio", default="tests/data/query_to_draw.wav")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()

    runner = Nemotron3_5ASRModelRunner(
        args.model_path,
        device="npu",
        enable_encoder_cuda_graph=False,
        encoder_graph_max_batch_size=4,
        dtype=args.dtype,
        num_lookahead_tokens=3,
    )
    runner.configure_encoder_state_pool(1)
    spec = runner.streaming_chunk_spec
    waveform, _ = librosa.load(args.audio, sr=16000)
    first_samples = int(spec["first_samples"])
    subsequent_samples = int(spec["subsequent_samples"])
    chunks = [waveform[:first_samples]]
    offset = first_samples
    while offset < len(waveform):
        chunk = waveform[offset : offset + subsequent_samples]
        if len(chunk) < subsequent_samples:
            chunk = np.pad(chunk, (0, subsequent_samples - len(chunk)))
        chunks.append(chunk)
        offset += subsequent_samples

    state = runner.new_streaming_decode_state()
    started_at_s = time.perf_counter()
    for index, chunk in enumerate(chunks):
        prepared = runner.prepare_streaming_chunk(
            chunk, language="auto", is_first=index == 0
        )
        result = runner.run_streaming_batch(
            [state], [prepared], requested_languages=["auto"]
        )
        print(
            f"chunk={index} elapsed_s={result.elapsed_s:.3f} tokens={len(state.tokens)} "
            f"text={result.clean_texts[0]!r}"
        )
    print(f"total_s={time.perf_counter() - started_at_s:.3f}")
    runner.encoder_state_pool.close()


if __name__ == "__main__":
    main()
