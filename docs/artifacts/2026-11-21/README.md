# Step 0 artifacts — 2026-11-21

Raw JSON from the follow-up pod, kept here because `logs/` is git-ignored and
the pod is ephemeral. Hardware/stack for every file: 1× Ascend 910B4 (A2,
32 GB HBM), driver 25.5.1, CANN 9.2.1, Python 3.11.18, torch 2.9.0+cpu /
torch_npu 2.9.0, sglang 0.5.23 (`pyproject_npu.toml`), sglang-omni
`feat/nemotron35-asr-npu` @ `ae62c57`, transformers 5.19.0, triton-ascend
3.2.0, 64 GiB host-memory cgroup.

| File | Produced by | Contents |
|---|---|---|
| `bench_fp32_c124.json` | `scripts/step0/bench_streaming.py --concurrency 1,2,4 --runs 3` | Per-concurrency first-window / steady median / p95 / wall / aggregate RTF; cold compile and peak host RSS. |
| `lifecycle_fp32.json` | `scripts/step0/realtime_lifecycle.py` | Warm server: slot reuse (A, blank B, A), interleaved sessions, disconnect/reconnect, socket-close cancellation. All pass. |
| `lifecycle_cold_cancel_fp32.json` | `scripts/step0/realtime_lifecycle.py --only cancel` on a fresh server | The cold-compile cancellation case: the follow-up session receives `session_open_timeout`; the server settles afterwards. Expected `passed: false` — it records the limitation. |
| `npu_graph_probe_fp32.json` | `scripts/step0/npu_graph_probe.py --capture-stream dedicated` | Whole-runner capture attempt; fails at `AclmdlRICaptureEnd` 107025. |
| `npu_graph_probe_fp32_current.json` | `scripts/step0/npu_graph_probe.py --capture-stream current` | Same failure with capture on the current stream. |
| `npu_capture_ops.json` | `scripts/step0/npu_capture_ops.py` | Op-level capture: linear/conv1d/SDPA/one_hot pass, LSTM (`DynamicRNN` aclop) fails. |
| `npu_capture_encoder.json` | `scripts/step0/npu_capture_encoder.py` (one target per process) | Module-level capture: conv/FFN/positions/projectors pass, SDPA attention fails, `eager` attention passes, full encoder then fails on `aclrtMemcpy`. |
| `parity_fixed_npu.json` | `scripts/step0/multilingual_parity.py --manifest /tmp/fleurs_audio/manifest.tsv --dtypes float32,bfloat16` | FLEURS en/de/ja/zh parity on the fixed vendored path: 16/16 transcripts, fp32 and bf16 exact on 11/16. |
| `parity_fixed_cpu_fp32.json` | `multilingual_parity_cpu.py --manifest /tmp/fleurs_audio/manifest.tsv` (HF path, torch_npu conversion disabled) | Same clips through the Transformers path on CPU. |

No model weights, audio, or server logs are stored here.
