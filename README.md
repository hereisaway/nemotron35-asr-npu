# Nemotron 3.5 ASR on Ascend NPU — workspace

Last updated: 2026-11-19

## Task

Bring `nvidia/nemotron-3.5-asr-streaming-0.6b` to Ascend NPU in SGLang-Omni, covering:

- offline transcription: `POST /v1/audio/transcriptions`
- native PCM streaming: `/v1/realtime`
- target hardware: 1× Ascend 910B3 (A2, 64 GB HBM), CANN 9.x

This is an **NPU-only** workstream. GPU (CUDA) support already exists in main;
XPU already has a recipe. Do not duplicate those paths.

## Issue and tracking

- Dedicated issue: https://github.com/sgl-project/sglang-omni/issues/4310
  - Title: `[NPU] Nemotron 3.5 ASR on Ascend roadmap`
  - Status: opened 2026-11-19, awaiting maintainer assignment
- Claim comment on NPU roadmap: https://github.com/sgl-project/sglang-omni/issues/1597#issuecomment-7013697894
  - Requested: add to #1597 Models checklist, assign
  - Pinged: @ping1jing2 (assignee/owner), @wangyao-i (Qwen3-ASR NPU conventions)
- Related:
  - Roadmap focus: https://github.com/sgl-project/sglang-omni/issues/1081
  - Precedent for NPU model tracking: https://github.com/sgl-project/sglang-omni/issues/2076

## Model

| Item | Value |
|---|---|
| HF repo | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Params | 637,997,088 (~0.64B, F32 checkpoint) |
| Architecture | multilingual RNN-T, greedy decoding only |
| Capabilities | offline transcription + native PCM streaming without VAD/manual commit |
| Memory | small; 910B3 64 GB and 5090 32 GB are both ample |

## Existing code to build on

- Model family: `sglang_omni/models/nemotron3_5_asr/`
  - `config.py`: single `asr` stage, `architecture = "Nemotron3_5AsrForRNNT"`
  - `stages.py`: `create_nemotron3_5_asr_executor`
  - `model_runner.py`: `GRAPH_CAPTURE_DEVICE_TYPES = frozenset({"cuda", "xpu"})`
    - NPU is not in the set → encoder graph capture is unavailable; eager fallback via
      `--asr.factory.enable_encoder_cuda_graph false`
  - `encoder.py`, `decoder.py`, `streaming.py`, `realtime.py`, `session.py`
- Docs:
  - `docs/cookbook/nemotron3_5_asr.md`
  - `docs/basic_usage/asr.md`
  - `docs/other_platforms/installation_xpu.md` (Nemotron 3.5 ASR, single XPU)
- NPU conventions:
  - `examples/configs/qwen3_tts_1_7b_npu.yaml` (attention_backend: ascend, sampling_backend: pytorch)
  - `sglang_omni/models/higgs_tts/npu_fallback.py` (pure-torch sampling fallbacks)
  - #2076 (Qwen3-ASR NPU: NPUGraph, BF16, A3 reference)

## Plan

### M0 — eager correctness on 910B3
- [ ] Pin environment: CANN 9.x, torch_npu version, sglang-omni commit
- [ ] Reproduce the GPU/XPU reference path (offline + `/v1/realtime`)
- [ ] Run eager on NPU with `--asr.factory.enable_encoder_cuda_graph false`
- [ ] Fix op/device/dtype issues (RNN-T decoder, conv encoder, streaming state)
- [ ] Compare BF16 transcripts against GPU on a small multilingual set

### M1 — performance
- [ ] Decide: extend graph capture to NPU (new backend) or document eager fallback
- [ ] Measure streaming window / latency / throughput at c1 and higher concurrency
- [ ] Record SKU explicitly (910B3/A2 vs #2076's A3 reference)

### M2 — coverage
- [ ] Add/extend unit tests if any are device-sensitive
- [ ] Ascend CI coverage if an NPU runner is available
- [ ] PR with results, limitations, and reproduction commands

## Risks / open questions

- CANN op coverage for RNN-T decoder / conv encoder / streaming state.
- NPU graph capture support (ATB/FIA) and its interaction with streaming buckets.
- 910B3 (A2) is not the A3 reference used in #2076; maintainers may ask for A3 results.
- NPU toolchain maturity (operator libs / drivers / build), noted in #1597 comments.

## Next actions

1. Wait for @ping1jing2 assignment / maintainer guidance.
2. Pin and record the exact CANN / torch_npu / sglang-omni versions.
3. Get the GPU/XPU reference passing, then attempt the NPU eager path.
4. Record all commands and logs under this directory.
