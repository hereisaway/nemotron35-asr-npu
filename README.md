# Nemotron 3.5 ASR on Ascend NPU — workspace

Last updated: 2026-11-21

## Task

Bring `nvidia/nemotron-3.5-asr-streaming-0.6b` to Ascend NPU in SGLang-Omni, covering:

- offline transcription: `POST /v1/audio/transcriptions`
- native PCM streaming: `/v1/realtime`
- target hardware: 1× Ascend 910B3 (A2, 64 GB HBM), CANN 9.x

This is an **NPU-only** workstream. GPU (CUDA) support already exists in main;
XPU already has a recipe. Do not duplicate those paths.

## Step 0 status (2026-11-21)

Eager NPU bring-up is complete on 1x Ascend 910B4 (A2, 32 GB HBM). It was first brought up under an 8 GiB host-memory cgroup, where server `/v1/realtime` could not finish first-window TBE compilation. On a follow-up pod with a 64 GiB host-memory cgroup, `/v1/realtime` now passes end to end (`final='What is happening in this video'`) and the offline and streaming eager paths still reproduce the Step 0 transcripts. Details and commands: [docs/step0-report.md](docs/step0-report.md). Reproduction scripts: `scripts/step0/`.

Pinned stack: CANN 9.2.1, Python 3.11.18, torch 2.9.0+cpu, torch_npu 2.9.0, sglang 0.5.23 (source install with `pyproject_npu.toml`), sglang-omni `3b73fdf38` on fork branch `feat/nemotron35-asr-npu`, transformers 5.19.0, triton-ascend 3.2.0.

Two code changes are on `feat/nemotron35-asr-npu` at `44d1a80ba`: NPU streaming must synchronize through `has_device_streams`, and the realtime adapter must use the shared audio-session `SessionLimits` (300 s operation timeout, 64 pending chunks) instead of the 30 s defaults. The NPU example config, docs and unit tests are in the same commit. Draft PR: [#4381](https://github.com/sgl-project/sglang-omni/pull/4381). `tests/unit_test/nemotron3_5_asr`: 58 passed, 5 skipped.

The official SGLang 0.5.23 NPU manifest targets CANN 9.1.0, Python 3.12 and torch/torch_npu 2.10; this pod's combination is a bring-up stack, not the upstream-validated one. Keep `TBE_PARALLEL_COMPILER=0`, `TBE_OP_PARALLEL_COMPILER=0` and `MAX_COMPILE_CORE_NUMBER=1` unless the kernel cache is prewarmed.

## Multilingual parity status (2026-11-21)

FLEURS en/de/ja/zh (4 clips each) was run for float32 and bfloat16 and is not usable as an M0 parity signal yet. The current HF/vendor inference path emits blank or unstable transcripts on most clips (13/16 blank on NPU). The same happens on CPU with the same code, with native Transformers 5.13 and 5.19, and with the model card's own `bcn_weather.mp3` example, so this is not NPU-specific. NPU and CPU also disagree on those unstable clips; float32 and bfloat16 agree on 15/16. Follow-up: compare against a NeMo / NeMo-Speech.cpp GGUF reference before claiming M0 correctness. The evaluation script is `scripts/step0/multilingual_parity.py`; JSON artifacts are written under `logs/` (git-ignored).

## Issue and tracking

- Dedicated issue: https://github.com/sgl-project/sglang-omni/issues/4310
  - Title: `[NPU] Nemotron 3.5 ASR on Ascend roadmap`
  - Status: open; **assigned to @ping1jing2 on 2026-11-19 17:35** (tracking owner, same pattern as #2076). Author @hereisaway is the implementer; co-assignee clarification is optional/pending.
- Claim comment on NPU roadmap: https://github.com/sgl-project/sglang-omni/issues/1597#issuecomment-7013697894
  - Requested: add to #1597 Models checklist, assign
  - Response: @ping1jing2 replied **"already done"** on 2026-11-19 17:40
  - #1597 checklist now contains: `- [ ] **Nemotron 3.5 ASR** (@hereisaway #4310 )`
  - Pinged: @ping1jing2 (assignee/owner), @wangyao-i (Qwen3-ASR NPU conventions)
- Draft PR: https://github.com/sgl-project/sglang-omni/pull/4381
  - Title: `[NPU] Nemotron 3.5 ASR: eager execution and realtime session limits`
  - Status: draft, open; one commit `44d1a80ba`, nine files.
- Step 0 status comment: https://github.com/sgl-project/sglang-omni/issues/4310#issuecomment-7038594567
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
- [x] Pin environment: CANN 9.x, torch_npu version, sglang-omni commit
- [ ] Reproduce the GPU/XPU reference path (offline + `/v1/realtime`) — no GPU in the pod; compare against the recorded XPU/CUDA transcripts
- [x] Run eager on NPU with `--asr.factory.enable_encoder_cuda_graph false`
- [x] Fix op/device/dtype issues (RNN-T decoder, conv encoder, streaming state) — no operator changes needed; PR 1 adds NPU stream synchronization and realtime limits
- [ ] Compare BF16 transcripts against GPU on a small multilingual set — blocked by HF/vendor blank outputs; needs a NeMo/GGUF reference
- [x] Finish server `/v1/realtime` validation on a host with more than the 8 GiB cgroup used in Step 0 — passed on a 64 GiB cgroup

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
- The test pod is 910B4 (A2, 32 GB), not the 910B3 target and not the A3 reference used in #2076; maintainers may ask for A3 results.
- NPU toolchain maturity (operator libs / drivers / build), noted in #1597 comments.
- The HF/vendor Nemotron inference path emits blank or unstable transcripts on most FLEURS clips, on CPU and NPU alike, including the model card's own `bcn_weather.mp3` example. Multilingual M0 parity needs a NeMo / NeMo-Speech.cpp reference first.

## Next actions

1. [done] Claim accepted; #1597 checklist updated with `Nemotron 3.5 ASR (@hereisaway #4310)`.
2. [done] Pin and record the exact CANN / torch_npu / sglang-omni versions.
3. [done] Run the NPU eager path and the in-process streaming path for float32 and bfloat16.
4. [done] Push PR 1 changes and open draft PR #4381.
5. [done] Start a pod with more host memory (64 GiB) and finish server `/v1/realtime`.
6. [done] Record commands and reproduction scripts under `docs/step0-report.md` and `scripts/step0/`.
7. Build a NeMo / NeMo-Speech.cpp (GGUF) reference and re-run en/de/ja/zh parity.
8. Add the native realtime lifecycle checks requested in #4310: interleaved sessions, disconnect/reconnect, cancellation during encoding, and slot reuse with different audio.
9. Ask @ping1jing2 to confirm whether 910B4 (A2) results are enough or 910B3 validation is needed.
10. Start M1: evaluate `NemotronStreamingEncoderGraphRunner` + `NpuDeviceGraphBackend` before changing NPU graph eligibility.
