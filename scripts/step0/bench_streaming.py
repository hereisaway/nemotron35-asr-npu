"""Measure eager streaming latency and throughput at c1..cN on Ascend NPU.

Reports, per concurrency: cold first-window time (includes TBE compilation),
steady-state window latency (median/p95), end-to-end realtime factor, and peak
host RSS. Meant for the M1 numbers in the Step 0 report; the SKU must be
recorded next to the results.

Usage:
  python bench_streaming.py --concurrency 1,2,4 --runs 3 \
      --json-out logs/bench_fp32_c124.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time

import librosa
import numpy as np
from sglang_omni.models.nemotron3_5_asr.model_runner import Nemotron3_5ASRModelRunner

DTYPE_BYTES = {"float32": 4, "bfloat16": 2}


def build_chunks(waveform: np.ndarray, spec: dict) -> list[np.ndarray]:
    first = int(spec["first_samples"])
    step = int(spec["subsequent_samples"])
    chunks = [waveform[:first]]
    offset = first
    while offset < len(waveform):
        chunk = waveform[offset : offset + step]
        if len(chunk) < step:
            chunk = np.pad(chunk, (0, step - len(chunk)))
        chunks.append(chunk)
        offset += step
    return chunks


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("NEMOTRON_MODEL_PATH"),
        required=os.environ.get("NEMOTRON_MODEL_PATH") is None,
    )
    parser.add_argument("--audio", default="tests/data/query_to_draw.wav")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="float32")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--lookahead-tokens", type=int, default=3)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    concurrencies = [int(value) for value in args.concurrency.split(",") if value]
    waveform, _ = librosa.load(args.audio, sr=16000)

    runner = Nemotron3_5ASRModelRunner(
        args.model_path,
        device="npu",
        enable_encoder_cuda_graph=False,
        encoder_graph_max_batch_size=max(concurrencies),
        dtype=args.dtype,
        num_lookahead_tokens=args.lookahead_tokens,
    )
    runner.configure_encoder_state_pool(max(concurrencies))
    spec = runner.streaming_chunk_spec
    chunks = build_chunks(waveform, spec)
    audio_seconds = len(waveform) / 16000.0

    report: dict = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "audio": args.audio,
        "audio_seconds": round(audio_seconds, 3),
        "chunks": len(chunks),
        "chunk_spec": {key: int(value) for key, value in spec.items()},
        "results": [],
    }

    cold_compile_s = None
    for concurrency in concurrencies:
        runs = []
        for run_index in range(args.runs):
            states = [runner.new_streaming_decode_state() for _ in range(concurrency)]
            windows: list[float] = []
            wall_started = time.perf_counter()
            try:
              for index, chunk in enumerate(chunks):
                prepared = [
                    runner.prepare_streaming_chunk(
                        chunk, language="auto", is_first=index == 0
                    )
                    for _ in range(concurrency)
                ]
                started = time.perf_counter()
                result = runner.run_streaming_batch(
                    states, prepared, requested_languages=["auto"] * concurrency
                )
                windows.append(time.perf_counter() - started)
                assert len(result.clean_texts) == concurrency
            finally:
                for state in states:
                    state.release_encoder_state()
            wall = time.perf_counter() - wall_started
            if cold_compile_s is None:
                cold_compile_s = windows[0]
            steady = windows[1:]
            runs.append(
                {
                    "first_window_s": round(windows[0], 4),
                    "steady_median_s": round(statistics.median(steady), 4),
                    "steady_p95_s": round(percentile(steady, 0.95), 4),
                    "steady_max_s": round(max(steady), 4),
                    "wall_s": round(wall, 3),
                    "realtime_factor": round(audio_seconds * concurrency / wall, 3),
                    "final_text": result.clean_texts[0],
                }
            )
        medians = {
            "first_window_s": round(statistics.median(r["first_window_s"] for r in runs), 4),
            "steady_median_s": round(statistics.median(r["steady_median_s"] for r in runs), 4),
            "steady_p95_s": round(statistics.median(r["steady_p95_s"] for r in runs), 4),
            "wall_s": round(statistics.median(r["wall_s"] for r in runs), 3),
            "realtime_factor": round(statistics.median(r["realtime_factor"] for r in runs), 3),
        }
        report["results"].append(
            {"concurrency": concurrency, "warm": medians, "runs": runs}
        )
        print(
            f"c={concurrency} first={medians['first_window_s']}s "
            f"steady_median={medians['steady_median_s']}s "
            f"p95={medians['steady_p95_s']}s wall={medians['wall_s']}s "
            f"rtf={medians['realtime_factor']}x "
            f"text={runs[-1]['final_text']!r}",
            flush=True,
        )

    runner.encoder_state_pool.close()
    report["cold_compile_s"] = round(cold_compile_s or 0.0, 4)
    report["peak_host_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1
    )
    payload = json.dumps(report, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
