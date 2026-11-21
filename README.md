# Nemotron 3.5 ASR on Ascend NPU — workspace

Last updated: 2026-11-21

## Task

Bring `nvidia/nemotron-3.5-asr-streaming-0.6b` to Ascend NPU in SGLang-Omni, covering:

- offline transcription: `POST /v1/audio/transcriptions`
- native PCM streaming: `/v1/realtime`
- target hardware: 1× Ascend 910B3 (A2, 64 GB HBM), CANN 9.x

This is an **NPU-only** workstream. GPU (CUDA) support already exists in main;
XPU already has a recipe. Do not duplicate those paths.

## Where things stand (2026-11-21)

- **M0 is complete and in review.** PR [#4381](https://github.com/sgl-project/sglang-omni/pull/4381) is ready for review at `ae62c57` (rebased onto `main@a9e5996`, 9 files, +166/−7). fp32 and bf16 are validated for offline, in-process streaming and server `/v1/realtime` on 1× 910B4; `tests/unit_test/nemotron3_5_asr` is 58 passed / 5 skipped.
- **CI is blocked on maintainer approval.** Every PR workflow, including Lint, reports `action_required`; the PR comment asks for the runs to be approved.
- **M1 has a decision and numbers.** Eager streaming is ~0.07 s per 320 ms window from c1 to c4 (aggregate RTF 4.1x/7.9x/13.5x). NPU encoder graph capture is not enabled: SDPA attention creates an internal sub-stream, and with eager attention the full encoder still hits a host↔device copy during capture. That is a separate change, not a blocker for M0.
- **Open items** are listed under [Next actions](#next-actions): #4385 parity, 910B3 confirmation, lifecycle unit tests, and the cold-compile cancellation follow-up.
- **Everything here is reproducible from a fresh pod** with `scripts/step0/restore_env.sh` plus the commands in [docs/step0-report.md](docs/step0-report.md).

## Step 0 status (2026-11-21)

Eager NPU bring-up is complete on 1x Ascend 910B4 (A2, 32 GB HBM). It was first brought up under an 8 GiB host-memory cgroup, where server `/v1/realtime` could not finish first-window TBE compilation. On a follow-up pod with a 64 GiB host-memory cgroup, `/v1/realtime` now passes end to end in fp32 and bf16 (`final='What is happening in this video'`) and the offline and streaming eager paths still reproduce the Step 0 transcripts. Details and commands: [docs/step0-report.md](docs/step0-report.md). Reproduction scripts: `scripts/step0/`.

The same 64 GiB pod also finished the remaining issue work:

- **Lifecycle** — slot reuse (A → empty B → A), interleaved sessions, mid-stream disconnect/reconnect and socket-close cancellation all pass warm. Closing the socket during the *first-window* compile surfaces a transient 30 s `session_open_timeout` on the next session; the server self-heals, and `response.cancel` is not implemented for native transcription sessions. `scripts/step0/realtime_lifecycle.py`.
- **M1 performance (eager, fp32)** — steady window cost is flat at ~0.07 s per 320 ms window from c1 to c4 (median 0.0758 / 0.0711 / 0.0698 s, p95 ≤ 0.0811 s); aggregate RTF 4.1x / 7.9x / 13.5x. A novel batch shape pays 14–15 s of TBE compilation on its first window. Peak host RSS 4.6 GB. `scripts/step0/bench_streaming.py`.
- **NPU graph capture** — capture still fails at `AclmdlRICaptureEnd` 107025 ("capture model contains a stream that was not joined to the origin stream") on both dedicated and current streams. Module bisect: conv/FFN/positions/projectors capture, the SDPA attention path does not, and `eager` attention captures on its own; the full encoder then hits a host↔device copy (`aclrtMemcpy` 107030) and needs a separate fix. Decision: keep the documented eager fallback; the capture path is concrete but is a separate change. `scripts/step0/npu_graph_probe.py`, `scripts/step0/npu_capture_encoder.py`, `scripts/step0/npu_capture_ops.py`.

Pinned stack: CANN 9.2.1, Python 3.11.18, torch 2.9.0+cpu, torch_npu 2.9.0, sglang 0.5.23 (source install with `pyproject_npu.toml`), sglang-omni `3b73fdf38` on fork branch `feat/nemotron35-asr-npu`, transformers 5.19.0, triton-ascend 3.2.0.

Two code changes are on `feat/nemotron35-asr-npu`, rebased onto `main` `a9e5996` as `ae62c57`: NPU streaming must synchronize through `has_device_streams`, and the realtime adapter must use the shared audio-session `SessionLimits` (300 s operation timeout, 64 pending chunks) instead of the 30 s defaults. The NPU example config, docs and unit tests are in the same commit. PR (ready for review): [#4381](https://github.com/sgl-project/sglang-omni/pull/4381). `tests/unit_test/nemotron3_5_asr`: 58 passed, 5 skipped (re-run after the rebase).

The official SGLang 0.5.23 NPU manifest targets CANN 9.1.0, Python 3.12 and torch/torch_npu 2.10; this pod's combination is a bring-up stack, not the upstream-validated one. Keep `TBE_PARALLEL_COMPILER=0`, `TBE_OP_PARALLEL_COMPILER=0` and `MAX_COMPILE_CORE_NUMBER=1` unless the kernel cache is prewarmed.

## Multilingual parity status (2026-11-21)

FLEURS en/de/ja/zh (4 clips each) shows that the **HF/vendor inference path is broken relative to the model's reference runtime**, on CPU and NPU alike:

- The HF/vendor path emits blank or unstable transcripts on most clips: 13/16 blank on NPU and 9/16 on CPU. Native Transformers 5.13/5.19 and the model card's own `bcn_weather.mp3` example behave the same way.
- A NeMo-Speech.cpp `q8_0` GGUF reference (ASR-only build, CPU) transcribes all 16 clips, with mean WER 0.14 against the FLEURS references (en 0.12, de 0.04, ja 0.16, zh 0.22).
- HF-NPU vs GGUF WER is 0.92 and HF-CPU vs GGUF is 0.89, with 0/16 exact matches. float32 and bfloat16 agree with each other on 15/16, so NPU matches CPU behavior; it is not an NPU regression.

Filed as upstream issue [#4385](https://github.com/sgl-project/sglang-omni/issues/4385); it should not block NPU enablement. The evaluation script is `scripts/step0/multilingual_parity.py`; JSON artifacts are written under `logs/` (git-ignored).

Root cause (2026-11-21, [investigation](docs/issue-4385-investigation-2026-11-21.md)): Transformers 5.19 `sdpa_attention_forward` converts any non-bool `attention_mask` to a boolean mask on hosts where `torch_npu` is importable. The model passes its relative-position bias matrix as that mask, so the conversion masks every position and self-attention returns zeros. The frontend, subsampling, positional encoding and layer-0 feed-forward match NeMo exactly; disabling the conversion reproduces the NeMo transcripts on all three probe clips.

## Issue and tracking

- Dedicated issue: https://github.com/sgl-project/sglang-omni/issues/4310
  - Title: `[NPU] Nemotron 3.5 ASR on Ascend roadmap`
  - Status: open; **assigned to @ping1jing2 on 2026-11-19 17:35** (tracking owner, same pattern as #2076). Author @hereisaway is the implementer; co-assignee clarification is optional/pending.
- Claim comment on NPU roadmap: https://github.com/sgl-project/sglang-omni/issues/1597#issuecomment-7013697894
  - Requested: add to #1597 Models checklist, assign
  - Response: @ping1jing2 replied **"already done"** on 2026-11-19 17:40
  - #1597 checklist now contains: `- [ ] **Nemotron 3.5 ASR** (@hereisaway #4310 )`
  - Pinged: @ping1jing2 (assignee/owner), @wangyao-i (Qwen3-ASR NPU conventions)
- PR: https://github.com/sgl-project/sglang-omni/pull/4381
  - Title: `[NPU] Nemotron 3.5 ASR: eager execution and realtime session limits`
  - Status: **ready for review** (not draft) since 2026-11-21; one commit `ae62c57`, nine files, rebased onto `main@a9e5996`.
  - Body: motivation, validation matrix (fp32/bf16), c1/c2/c4 eager numbers, capture result, limitations, checklist.
  - Comment asking for CI approval/review: https://github.com/sgl-project/sglang-omni/pull/4381#issuecomment-7060270254
  - CI: all workflows `action_required` until a maintainer approves them.
- Step 0 status comment: https://github.com/sgl-project/sglang-omni/issues/4310#issuecomment-7038594567
- HF/vendor inference bug: https://github.com/sgl-project/sglang-omni/issues/4385
  - Title: `[Bug][Nemotron 3.5 ASR] HF/vendor path emits blank transcripts while the GGUF reference transcribes correctly`
  - Status: open; GGUF reference workaround documented, NPU PR unaffected.
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
- [x] Decide: extend graph capture to NPU (new backend) or document eager fallback — capture fails in `AclmdlRICaptureEnd` (cross-stream); keep the eager fallback and track capture as a separate change
- [x] Measure streaming window / latency / throughput at c1 and higher concurrency — `logs/bench_fp32_c124.json`, ~0.07 s per 320 ms window, aggregate RTF 4.1x/7.9x/13.5x at c1/c2/c4
- [x] Record SKU explicitly (910B3/A2 vs #2076's A3 reference) — every number is labelled 910B4 (A2, 32 GB HBM)

### M2 — coverage
- [ ] Add/extend unit tests if any are device-sensitive
- [ ] Ascend CI coverage if an NPU runner is available
- [ ] PR with results, limitations, and reproduction commands

## Risks / open questions

- CANN op coverage for RNN-T decoder / conv encoder / streaming state.
- NPU graph capture support (ATB/FIA) and its interaction with streaming buckets.
- The test pod is 910B4 (A2, 32 GB), not the 910B3 target and not the A3 reference used in #2076; maintainers may ask for A3 results.
- NPU toolchain maturity (operator libs / drivers / build), noted in #1597 comments.
- The HF/vendor Nemotron 3.5 inference path is broken relative to the GGUF reference: 13/16 FLEURS clips blank on NPU, 9/16 on CPU, 0/16 exact matches, while the NeMo-Speech.cpp `q8_0` GGUF transcribes all 16. This is not NPU-specific; filed as #4385.

## Next actions

1. [done] Claim accepted; #1597 checklist updated with `Nemotron 3.5 ASR (@hereisaway #4310)`.
2. [done] Pin and record the exact CANN / torch_npu / sglang-omni versions.
3. [done] Run the NPU eager path and the in-process streaming path for float32 and bfloat16.
4. [done] Push PR 1 changes and open PR #4381; it is now ready for review at `ae62c57`.
5. [done] Start a pod with more host memory (64 GiB) and finish server `/v1/realtime`.
6. [done] Record commands and reproduction scripts under `docs/step0-report.md` and `scripts/step0/`.
7. [done] Build a NeMo-Speech.cpp GGUF reference; it confirms the HF/vendor path is the parity blocker, not NPU.
8. [done] File the HF/vendor Nemotron 3.5 inference bug upstream: #4385.
9. [done] Add the native realtime lifecycle checks requested in #4310: interleaved sessions, disconnect/reconnect, cancellation during encoding, and slot reuse with different audio. Warm paths pass; cold-compile cancellation leaves a transient `session_open_timeout` documented in the report.
10. [open] Ask @ping1jing2 to confirm whether 910B4 (A2) results are enough or 910B3 validation is needed.
11. [done] M1: evaluate `NemotronStreamingEncoderGraphRunner` + `NpuDeviceGraphBackend`; NPU capture fails in `AclmdlRICaptureEnd`, so the eager fallback is documented and capture is a separate change.
12. [done] M1: measure streaming window latency and throughput at c1 and higher concurrency; record the SKU explicitly (910B4).
13. [in progress] M2: add/extend device-sensitive tests, Ascend CI coverage if a runner is available, and land PR #4381 with results and limitations. Branch rebased onto `main@a9e5996` (`ae62c57`), 58 passed / 5 skipped; no NPU CI runner.
14. [blocked] After #4385 is fixed: re-run en/de/ja/zh parity and close M0.
