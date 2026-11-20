# Step 0 — NPU eager bring-up report

Date: 2026-11-20
Code under test: `sgl-project/sglang-omni` `3b73fdf38`, fork branch `feat/nemotron35-asr-npu`
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
| sglang-omni | 0.2.0, editable at `3b73fdf38` |
| transformers | 5.19.0 |
| triton-ascend | 3.2.0 |
| torchaudio / torchvision | 2.9.0 / 0.24.0 |
| numpy / pandas / soundfile | 1.26.4 / 2.2.3 / 0.13.1 |

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

## Findings

- TBE compiles the model's kernels on first use in each process, taking about 20 s for the streaming first window. Later windows in the same process are fast.
- On a 192-core host, TBE parallel compilation can spawn enough workers to exceed a small memory cgroup. `TBE_PARALLEL_COMPILER=0`, `TBE_OP_PARALLEL_COMPILER=0` and `MAX_COMPILE_CORE_NUMBER=1` keep the Pod alive for single-process tests.
- Even serialized, the server plus stage plus TBE compilation exceeds the 8 GiB cgroup. A larger host memory limit or a prewarmed kernel cache is required to finish server realtime validation.
- `Nemotron3_5ASRModelRunner` used `supports_device_streams`, which deliberately excludes NPU, so the streaming path skipped device synchronization. Tokens were still correct, but the measured window time was underreported. The fix is to use `has_device_streams`, matching the ARK-ASR NPU enablement.
- The realtime adapter for this model defaulted to `SessionLimits()` (30 s operation timeout, 16 pending chunks). A first NPU window can exceed 30 s once server startup overhead is included. The fix is to pass the realtime audio-session limits used by other native session models: 300 s, 64 pending chunks.

## Reproduction

Environment setup:

```bash
git clone https://github.com/hereisaway/sglang-omni.git
cd sglang-omni
git switch feat/nemotron35-asr-npu

git clone --depth 1 --branch v0.5.23 https://github.com/sgl-project/sglang.git /opt/atomgit/sglang
cd /opt/atomgit/sglang
cp python/pyproject_npu.toml python/pyproject.toml
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.23 pip install -e python
```

The pod environment was prepared with the Huawei PyPI mirror. Python dependencies for `sglang-omni` came from `pyproject_npu.toml` with torch, torch_npu, numpy and soundfile pinned to the versions above.

Model download from the Hugging Face mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  nvidia/nemotron-3.5-asr-streaming-0.6b \
  --local-dir /opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b \
  config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json model.safetensors
```

Offline, streaming and realtime client scripts are in `scripts/step0/`. Run them from the `sglang-omni` repository root with the model path passed through `NEMOTRON_MODEL_PATH`.

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

1. Land the `has_device_streams` synchronization fix and the realtime session-limit fix in PR 1.
2. Re-run server `/v1/realtime` on a host with more than 8 GiB memory, or after establishing a reusable CANN kernel cache.
3. Run multilingual parity (en/zh/de/ja) for float32 and bfloat16 before claiming M0 correctness.
4. Add the NPU example config, cookbook section and installation note.
