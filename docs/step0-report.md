# Step 0 — NPU eager bring-up report

Date: 2026-11-20 (initial pod), 2026-11-21 (follow-up pod)
Code under test: `sgl-project/sglang-omni` `3b73fdf38` (base), fork branch `feat/nemotron35-asr-npu` `44d1a80ba` (PR 1 changes, draft PR #4381)
Model: `nvidia/nemotron-3.5-asr-streaming-0.6b`

## Hardware and software

| Item | Value |
|---|---|
| NPU | 1x Ascend 910B4 (A2), 32 GB HBM, driver 25.5.1 |
| Host memory cgroup | 8 GiB |
| CANN | 9.2.1 |
| Python | 3.11.18 |
| torch / torch_npu | 2.9.0+cpu / 2.9.0 |
| sglang | 0.5.23 (built from source with `pyproject_npu.toml`) |
| sglang-omni | 0.2.0, editable at base `3b73fdf38`; PR 1 commit `44d1a80ba` |
| transformers | 5.19.0 |
| triton-ascend | 3.2.0 |
| torchaudio / torchvision | 2.9.0 / 0.24.0 |
| numpy / pandas / soundfile | 1.26.4 / 2.2.3 / 0.13.1 |

The follow-up pod (2026-11-21) used the same 910B4, CANN, Python and torch/torch_npu versions, but a 64 GiB host-memory cgroup instead of 8 GiB.

The official SGLang 0.5.23 NPU manifest targets CANN 9.1.0, Python 3.12 and torch/torch_npu 2.10. This pod runs the combinations above; they work for this model but are not the validated upstream stack.

## Results

| Path | Dtype | Result | Notes |
|---|---|---|---|
| Direct `model.generate()` | float32 | Pass | `What is happening in this videos are.` on `query_to_draw.wav` (5.06 s), 28.4 s |
| Direct `model.generate()` | bfloat16 | Pass | `What is happening video?`, 21.7 s; single-clip parity differs from float32 |
| `Nemotron3_5ASRModelRunner` streaming | float32 | Pass | `What is happening in this video`; first window 20.2 s, later windows ~0.07 s |
| `Nemotron3_5ASRModelRunner` streaming | bfloat16 | Pass | Same transcript; first window 21.0 s |
| Server `POST /v1/audio/transcriptions` | float32 | Pass | HTTP 200, 41.8 s including first-request compilation |
| Server `/v1/realtime` | float32 | Blocked by pod memory | WebSocket and session setup pass; first streaming window triggers TBE compilation and the 8 GiB host cgroup OOM-kills the ASR stage |

The model path does not need custom AscendC operators. Conv2d/Conv1d, LSTM, SDPA, softmax/argmax, indexing and casting all execute on CANN through `torch_npu`.

## Follow-up run (2026-11-21)

Same 910B4 (A2, 32 GB HBM), same pins, with a 64 GiB host-memory cgroup. `sglang-omni` was installed editable from `feat/nemotron35-asr-npu` `44d1a80ba`.

| Path | Dtype | Result | Notes |
|---|---|---|---|
| Direct `model.generate()` | float32 | Pass | `What is happening in this videos are.`, 16.1 s |
| Direct `model.generate()` | bfloat16 | Pass | `What is happening video?`, 42.3 s |
| `Nemotron3_5ASRModelRunner` streaming | float32 | Pass | `What is happening in this video`; first window ~16 s, later windows ~0.07 s |
| `Nemotron3_5ASRModelRunner` streaming | bfloat16 | Pass | Same transcript; total 16.5 s |
| Server `/v1/realtime` | float32 | Pass | Session and first window complete; `final='What is happening in this video'`, no OOM |
| Server `/v1/audio/transcriptions` | bfloat16 | Pass | `What is happening video?` (re-validated 2026-11-21) |
| Server `/v1/realtime` | bfloat16 | Pass | Three sessions, all `final='What is happening in this video'` (re-validated 2026-11-21) |
| `tests/unit_test/nemotron3_5_asr` | - | Pass | 58 passed, 5 skipped |

The server run used `--asr.factory.enable_encoder_cuda_graph false --enable-realtime` with serial TBE compilation and was healthy about 55 s after launch. The WebSocket client sent 16 kHz mono PCM16 in 640-byte chunks, waited 30 s for the first-window compile, then resumed and ended with `sglang.input_audio.end`.

### Multilingual parity: HF path vs GGUF reference

FLEURS en/de/ja/zh, 4 clips per locale, float32 and bfloat16, via `scripts/step0/multilingual_parity.py`. A NeMo-Speech.cpp `q8_0` GGUF reference (`nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`, ASR-only CPU build) was run on the same 16 clips.

| Locale | GGUF WER vs FLEURS | HF-NPU blank | HF-NPU WER vs GGUF | HF-CPU WER vs GGUF |
|---|---|---|---|---|
| en_us | 0.121 | 3/4 | 0.939 | 1.000 |
| de_de | 0.044 | 3/4 | 0.893 | 0.917 |
| ja_jp | 0.163 | 3/4 | 0.851 | 0.665 |
| cmn_hans_cn | 0.216 | 4/4 | 1.000 | 0.991 |
| overall | 0.136 | 13/16 | 0.921 | 0.893 |

`ja_jp` and `cmn_hans_cn` are scored character-level (CER).

- The GGUF reference transcribes every clip and is close to the FLEURS ground truth, so the audio and checkpoint are fine.
- The HF/vendor path blanks or degrades most clips: 13/16 blank on NPU, 9/16 on CPU; 0/16 exact matches against GGUF.
- float32 and bfloat16 agree on 15/16, and NPU and CPU fail on the same inputs, so this is not an NPU regression. The same behavior appears with native Transformers 5.13/5.19 and with the model card's `bcn_weather.mp3` example.
- Conclusion: the HF/vendor Nemotron 3.5 inference path is broken relative to the reference runtime. Filed as #4385; it should not block NPU enablement.

JSON artifacts are written under `logs/` and are git-ignored. Reference build: CMake 4.4.5 from pip, SentencePiece 0.2.1 static from the PyPI sdist into `.deps/sentencepiece`, llama.cpp submodule `bd4f514`, configured ASR-only and run with `--backend cpu`:

```bash
python -m pip install --target /tmp/cmake_pkg "cmake>=3.26"
export PYTHONPATH=/tmp/cmake_pkg
CMAKE="python -m cmake"
# SentencePiece 0.2.1 static from the PyPI sdist (source is in the sentencepiece/ subdir)
$CMAKE -G Ninja -S sentencepiece -B /tmp/sp_build -DCMAKE_BUILD_TYPE=Release \
  -DSPM_BUILD_TEST=OFF -DSPM_ENABLE_SHARED=OFF
$CMAKE --build /tmp/sp_build --target sentencepiece-static -j 16
# install libsentencepiece.a + sentencepiece_processor.h under .deps/sentencepiece/{lib,include}
$CMAKE -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DNEMO_SPEECH_BUILD_ASR=ON -DNEMO_SPEECH_BUILD_DIAR=OFF -DNEMO_SPEECH_BUILD_TTS=OFF \
  -DNEMO_SPEECH_BUILD_NMT=OFF -DNEMO_SPEECH_BUILD_S2S=OFF -DNEMO_SPEECH_BUILD_SPEECHLM=OFF \
  -DNEMO_SPEECH_BUILD_CLI=ON -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF -DNEMO_SPEECH_BUILD_HTTP=OFF \
  -DNEMO_SPEECH_BUILD_GRPC=OFF
$CMAKE --build build -j 32
./build/bin/nemo-speech transcribe clip.wav -m <model.q8_0.gguf> -l en-US --backend cpu -f text
```

## Realtime lifecycle checks (2026-11-21)

Run against the server (`--enable-realtime`, fp32, 64 GiB host cgroup) with
`scripts/step0/realtime_lifecycle.py`. The 16 kHz-only audio that transcribes
on the vendor/HF path is `tests/data/query_to_draw.wav`; every other clip in
`tests/data` returns an empty transcript (#4385), so the "blank" session uses
a resampled `query_to_cars.wav` and is expected to stay empty.

| Scenario | Result | Evidence |
|---|---|---|
| Slot reuse A → blank B → A | pass | A repeats exactly (`What is happening in this video`); the empty session does not leak encoder state |
| Interleaved sessions (A + blank B) | pass | the good stream matches the single-session reference; no cross-talk |
| Disconnect mid-stream, then reconnect | pass | recovered session transcribes correctly; no leaked stage session |
| Close socket during a window | pass (warm) | follow-up session transcribes correctly; `/health` settles to zero requests |
| Close socket during the *first-window* compile (cold) | **fails transiently** | next `session.update` trips `session_open_timeout` after 30 s; the server self-heals and the following session works |

Details on the cold-cancel failure:

- Closing the socket aborts the coordinator request immediately
  (`Coordinator aborted req=...`), but the stage is inside an uninterruptible
  TBE compilation for that window. The next session's admission queues behind
  it and exceeds the 30 s session-open deadline (`DEFAULT_OPERATION_TIMEOUT_S`),
  so the server answers `{"code": "session_open_timeout"}`.
- `--realtime-session-update-timeout-s 300` does **not** change this: that flag
  bounds the wait for the first `session.update`, not stage admission.
- `response.cancel` is not implemented for native transcription sessions; the
  server returns `not_supported` (`this session does not cancel responses`).
  Cancellation has to be expressed by closing the socket.
- This is a cold-start-only edge case (it needs an aborted compile in flight).
  Documenting it is enough for the eager PR; a follow-up can either honour the
  session's `operation_timeout_s` on the stage side or skip admission retries
  while a compile is running.

## M1 eager performance (2026-11-21)

Single process, fp32, `Nemotron3_5ASRModelRunner`, 5.06 s of audio (15 windows
of 320 ms), 910B4 (A2, 32 GB HBM), `scripts/step0/bench_streaming.py`
(`logs/bench_fp32_c124.json`):

| Concurrency | First window | Steady median | Steady p95 | Wall | Aggregate RTF |
|---|---|---|---|---|---|
| c1 | 14.37 s (cold compile) | 0.0758 s | 0.0811 s | 1.23 s | 4.1x |
| c2 | 0.58 s (new batch shape) | 0.0711 s | 0.0775 s | 1.28 s | 7.9x |
| c4 | 0.082 s (cache warm) | 0.0698 s | 0.0762 s | 1.50 s | 13.5x |

- Steady-state window cost is flat at ~0.07 s per 320 ms window from c1 to c4;
  batching four streams costs ~1.2x the single-stream wall time.
- The first window of a novel shape pays 14–15 s of TBE compilation; the
  second and later runs in the same process are warm.
- Peak host RSS for the in-process runner, including weights, was 4.6 GB.
- These are eager numbers: encoder graph capture is still unavailable on NPU.

## NPU graph capture probe (2026-11-21)

The platform already returns `NpuDeviceGraphBackend`, but
`GRAPH_CAPTURE_DEVICE_TYPES` only lists cuda/xpu. `scripts/step0/npu_graph_probe.py`
adds `npu` at runtime and builds the runner with
`enable_encoder_cuda_graph=True`:

| Attempt | Result |
|---|---|
| Capture on the runner's dedicated stream | fail in 15.5 s |
| Capture on the current stream | fail in 15.8 s |
| Error | `AclmdlRICaptureEnd` 107025: `capture model contains a stream that was not joined to the origin stream` |

Module-level bisect (`scripts/step0/npu_capture_encoder.py`, one process per
target because a failed capture poisons the process):

| Target | Capture |
|---|---|
| subsampling (Conv2d stack), `encode_positions`, `encoder_projector`, `joint`, `feed_forward1`, convolution module | pass |
| self-attention, default/sdpa | **fail** (107025, stream not joined) |
| self-attention, `eager` | pass |
| full encoder, default attention | fail (107025) |
| full encoder, `eager` attention | fail at `aclrtMemcpy` 107030 (host↔device copy inside capture) |

So the blocker is **not** the LSTM: the decoder prediction network is outside
the captured encoder region and is never captured. The encoder-side failures
are (1) the SDPA attention path creating an internal sub-stream, which `eager`
attention avoids, and (2) a host-side synchronization or scalar read inside the
full encoder once attention is eager. The conv stack, FFN, positional encoding
and projectors all capture; the earlier `DynamicRNN` finding only rules out
capturing the decoder step as-is.

Decision for M1: keep the documented eager fallback for this PR. Enabling NPU
capture is a separate change with a concrete path: capture with `eager`
attention and remove the remaining host↔device copy (or make it
stream-safe) inside the encoder before capture. The eager numbers below are
the baseline.

## Findings

- TBE compiles the model's kernels on first use in each process, taking about 20 s for the streaming first window. Later windows in the same process are fast.
- On a 192-core host, TBE parallel compilation can spawn enough workers to exceed a small memory cgroup. `TBE_PARALLEL_COMPILER=0`, `TBE_OP_PARALLEL_COMPILER=0` and `MAX_COMPILE_CORE_NUMBER=1` keep the Pod alive for single-process tests.
- Even serialized, the server plus stage plus TBE compilation exceeds the 8 GiB cgroup. A larger host memory limit is required to finish server realtime validation; a 64 GiB cgroup was validated on 2026-11-21.
- `Nemotron3_5ASRModelRunner` used `supports_device_streams`, which deliberately excludes NPU, so the streaming path skipped device synchronization. Tokens were still correct, but the measured window time was underreported. The fix is to use `has_device_streams`, matching the ARK-ASR NPU enablement.
- The realtime adapter for this model defaulted to `SessionLimits()` (30 s operation timeout, 16 pending chunks). A first NPU window can exceed 30 s once server startup overhead is included. The fix is to pass the realtime audio-session limits used by other native session models: 300 s, 64 pending chunks.

## Pod sizing

| Host memory cgroup | Outcome |
|---|---|
| 8 GiB | Fails; initial Step 0 pod |
| 16 GiB | Expected minimum with serial TBE compilation; not yet validated |
| 32 GiB | Recommended; not yet validated |
| 64 GiB | Validated on 2026-11-21; server `/v1/realtime` passes |

NPU memory does not need to change: 32 GB HBM is ample for a 0.64B model in float32 or bfloat16. Reserve at least 10 GB of disk for the model, logs and a possible compile cache.

Keep these settings unless TBE compilation has been prewarmed:

```bash
export TBE_PARALLEL_COMPILER=0
export TBE_OP_PARALLEL_COMPILER=0
export MAX_COMPILE_CORE_NUMBER=1
```

If the next pod uses the official SGLang 0.5.23 NPU image (CANN 9.1.0, Python 3.12, torch/torch_npu 2.10), prefer that validated stack and adjust only the sglang-omni install to `scripts/npu/install_npu.sh`.

## Handoff state

- `hereisaway/sglang-omni` branch `feat/nemotron35-asr-npu` is rebased onto `main@a9e5996` locally as `ae62c57` (still to push). It contains the `has_device_streams` synchronization fix, the realtime session-limit fix, the NPU example config, cookbook/installation notes and unit tests. Draft PR #4381 is open; 58 passed / 5 skipped after the rebase.
- `hereisaway/nemotron35-asr-npu` `main` contains this report, the README status, `scripts/step0/` and the multilingual parity script.
- Issue #4310 was updated on 2026-11-21 with the Step 0 results, the realtime pass and the parity blocker: https://github.com/sgl-project/sglang-omni/issues/4310#issuecomment-7038594567
- The HF/vendor inference bug is filed as #4385: https://github.com/sgl-project/sglang-omni/issues/4385. The GGUF reference confirms the audio and checkpoint are fine; NPU matches CPU behavior, so this does not block the NPU PR.

Remaining work, in order:

1. [done] Build a NeMo-Speech.cpp GGUF reference; it confirms the HF/vendor path is the parity blocker, not NPU.
2. [done] File the HF/vendor Nemotron 3.5 inference bug upstream: #4385. Re-run parity after a fix.
3. [done] Add the native realtime lifecycle checks requested in #4310; see the lifecycle section above (cold-compile cancellation is documented as a transient limitation).
4. [open] Confirm with @ping1jing2 whether 910B4 (A2) evidence is sufficient or 910B3 is required.
5. [done] M1: evaluate `NemotronStreamingEncoderGraphRunner` + `NpuDeviceGraphBackend`; capture fails in `AclmdlRICaptureEnd`, eager fallback documented.
6. [done] M1: measure streaming window latency and throughput at c1/c2/c4; SKU 910B4 recorded.
7. [in progress] M2: add/extend device-sensitive tests, Ascend CI coverage if a runner is available, and land PR #4381 with results and limitations.

## Reproduction

Clone the fork branch first:

```bash
git clone https://github.com/hereisaway/sglang-omni.git /opt/atomgit/sglang-omni
cd /opt/atomgit/sglang-omni
git switch -c feat/nemotron35-asr-npu origin/feat/nemotron35-asr-npu
```

The pod already provided torch 2.9.0+cpu, torch_npu 2.9.0 and CANN 9.2.1. The remaining install used the Huawei PyPI mirror and these pins:

```bash
# Pod-specific: the preinstalled venv was root-owned.
sudo chown -R atomgit:atomgitgroup /opt/venv
pip install -U "huggingface_hub[hf_xet]" hf_transfer

# sglang 0.5.23 from source with the NPU manifest.
git clone --depth 1 --branch v0.5.23 https://github.com/sgl-project/sglang.git /opt/atomgit/sglang
cd /opt/atomgit/sglang
cp python/pyproject_npu.toml python/pyproject.toml
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.23 pip install -e python

# sglang-omni dependencies from pyproject_npu.toml, with the base stack pinned.
cd /opt/atomgit/sglang-omni
python - <<'PY' > /tmp/omnireq.txt
import tomllib
d = tomllib.load(open("pyproject_npu.toml", "rb"))
print("\n".join(d["project"]["dependencies"]))
PY
printf '%s\n' \
  'torch==2.9.0' 'torch_npu==2.9.0' \
  'numpy==1.26.4' 'pandas==2.2.3' 'soundfile==0.13.1' \
  'torchaudio==2.9.0' 'torchvision==0.24.0' > /tmp/constraints.txt
pip install -r /tmp/omnireq.txt -c /tmp/constraints.txt
pip install "triton-ascend==3.2.0" pybind11 "torchaudio==2.9.0" pytest pytest-asyncio ruff jiwer -c /tmp/constraints.txt
pip install -e . --no-deps
```

For a clean pod, prefer installing `sglang-omni` against `pyproject_npu.toml` instead of the default CUDA manifest so package metadata matches the NPU stack. This pod used the default manifest with `--no-deps`; runtime behavior is the same, but `pip check` reports the CUDA torch pins as conflicts.

Model download from the Hugging Face mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  nvidia/nemotron-3.5-asr-streaming-0.6b \
  --local-dir /opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b \
  config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json model.safetensors
```

Offline, streaming and realtime client scripts are in `scripts/step0/`. Run them from the `sglang-omni` repository root with the model path passed through `NEMOTRON_MODEL_PATH`.

The multilingual parity script also needs `jiwer` and a manifest TSV of `lang<TAB>wav<TAB>reference` rows. The Step 0 run built it from the FLEURS `test.tsv` and `test.tar.gz` for `en_us`, `de_de`, `ja_jp` and `cmn_hans_cn`, then ran it with `--manifest /tmp/fleurs_audio/manifest.tsv`.

```bash
export NEMOTRON_MODEL_PATH=/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b
export TBE_PARALLEL_COMPILER=0 TBE_OP_PARALLEL_COMPILER=0 MAX_COMPILE_CORE_NUMBER=1
python /path/to/nemotron35-asr-npu/scripts/step0/offline_generate.py --dtype float32
python /path/to/nemotron35-asr-npu/scripts/step0/streaming_eager.py --dtype bfloat16
```

Server offline check:

```bash
sgl-omni serve --model-path "$NEMOTRON_MODEL_PATH" --host 127.0.0.1 --port 8011 \
  --asr.factory.dtype float32 --asr.factory.enable_encoder_cuda_graph false --enable-realtime
curl -sS -X POST http://127.0.0.1:8011/v1/audio/transcriptions \
  -F file=@tests/data/query_to_draw.wav -F model=nemotron-local -F response_format=verbose_json
python /path/to/nemotron35-asr-npu/scripts/step0/realtime_client.py --port 8011
```

## Next

1. Land PR #4381 (the `has_device_streams` synchronization fix and the realtime session-limit fix).
2. Track #4385 and re-run multilingual parity after a fix.
3. Add the native realtime lifecycle checks requested in #4310.
4. M1: evaluate NPU encoder graph capture, otherwise document the eager fallback.
5. M1: measure streaming latency and throughput at c1 and higher concurrency; record the SKU.
6. M2: add device-sensitive tests, Ascend CI coverage if a runner is available, and finalize PR results/limitations.
