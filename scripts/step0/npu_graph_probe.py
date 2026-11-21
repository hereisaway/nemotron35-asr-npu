"""Probe NPU graph capture for the Nemotron 3.5 streaming encoder.

The model gates graph capture on ``GRAPH_CAPTURE_DEVICE_TYPES``, which lists
cuda/xpu only, and the platform already provides ``NpuDeviceGraphBackend``.
This script adds "npu" at runtime, captures the encoder, and reports either a
working capture (with timings) or the exact failure to document the eager
fallback.

Usage:
  PER_STREAM_QUEUE=1 TASK_QUEUE_ENABLE=1 python npu_graph_probe.py \
      --model-path "$NEMOTRON_MODEL_PATH" --json-out logs/npu_graph_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from contextlib import contextmanager

os.environ.setdefault("PER_STREAM_QUEUE", "1")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.pop("ASCEND_LAUNCH_BLOCKING", None)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch_npu  # noqa: F401,E402

import sglang_omni.models.nemotron3_5_asr.model_runner as model_runner  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("NEMOTRON_MODEL_PATH"),
        required=os.environ.get("NEMOTRON_MODEL_PATH") is None,
    )
    parser.add_argument("--audio", default="tests/data/query_to_draw.wav")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--lookahead-tokens", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument(
        "--capture-stream",
        choices=["dedicated", "current"],
        default="dedicated",
        help="capture on the runner's dedicated stream, or on the current stream",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    report: dict = {
        "device": "npu",
        "dtype": args.dtype,
        "max_batch_size": args.max_batch_size,
        "capture_stream": args.capture_stream,
        "device_types_before": sorted(model_runner.GRAPH_CAPTURE_DEVICE_TYPES),
    }

    model_runner.GRAPH_CAPTURE_DEVICE_TYPES = frozenset(
        set(model_runner.GRAPH_CAPTURE_DEVICE_TYPES) | {"npu"}
    )

    if args.capture_stream == "current":
        from sglang_omni.platforms import device_graph

        backend_capture = device_graph.NpuDeviceGraphBackend.capture

        @contextmanager
        def capture_on_current_stream(self, *, pool=None, stream=None, thread_local_errors=False):
            with backend_capture(
                self, pool=pool, stream=None, thread_local_errors=thread_local_errors
            ) as graph:
                yield graph

        device_graph.NpuDeviceGraphBackend.capture = capture_on_current_stream
    report["device_types_after"] = sorted(model_runner.GRAPH_CAPTURE_DEVICE_TYPES)

    runner = model_runner.Nemotron3_5ASRModelRunner(
        args.model_path,
        device="npu",
        enable_encoder_cuda_graph=True,
        encoder_graph_max_batch_size=args.max_batch_size,
        dtype=args.dtype,
        num_lookahead_tokens=args.lookahead_tokens,
    )
    report["backend"] = type(runner.graph_backend).__name__
    report["graph_enabled"] = bool(runner.enable_encoder_cuda_graph)

    started = time.perf_counter()
    try:
        runner.configure_encoder_state_pool(1)
    except Exception as error:  # noqa: BLE001 - the failure is the result
        report.update(
            {
                "captured": False,
                "capture_s": round(time.perf_counter() - started, 3),
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            }
        )
    else:
        report.update(
            {
                "captured": True,
                "capture_s": round(time.perf_counter() - started, 3),
                "captured_batches": sorted(runner.encoder_graph_runner.captured_batches),
            }
        )
        waveform, _ = librosa.load(args.audio, sr=16000)
        chunks = build_chunks(waveform, runner.streaming_chunk_spec)
        state = runner.new_streaming_decode_state()
        windows: list[float] = []
        text = ""
        for index, chunk in enumerate(chunks[: args.windows]):
            prepared = runner.prepare_streaming_chunk(
                chunk, language="auto", is_first=index == 0
            )
            begin = time.perf_counter()
            result = runner.run_streaming_batch(
                [state], [prepared], requested_languages=["auto"]
            )
            windows.append(round(time.perf_counter() - begin, 4))
            text = result.clean_texts[0]
        report["windows_s"] = windows
        report["text"] = text
    finally:
        try:
            runner.close()
        except Exception:  # noqa: BLE001
            pass

    payload = json.dumps(report, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    if not report["captured"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
