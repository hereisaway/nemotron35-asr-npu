# Step 0 scripts

Run from an `sglang-omni` checkout with `NEMOTRON_MODEL_PATH` set to the local
snapshot of `nvidia/nemotron-3.5-asr-streaming-0.6b`.

| Script | Purpose |
|---|---|
| `restore_env.sh` | Idempotent environment install: sglang 0.5.23 (`pyproject_npu.toml`), torch/torch_npu re-pin, triton-ascend, sglang-omni fork branch, model download, unit tests. |
| `offline_generate.py` | One `model.generate()` transcription. |
| `streaming_eager.py` | In-process eager streaming through `Nemotron3_5ASRModelRunner`. |
| `realtime_client.py` | One WebSocket session against `/v1/realtime`. |
| `realtime_lifecycle.py` | Slot reuse, interleaved sessions, disconnect/reconnect and cancellation checks. Writes JSON. |
| `bench_streaming.py` | Window latency / throughput at c1..cN. Writes JSON. |
| `npu_graph_probe.py` | Adds `npu` to `GRAPH_CAPTURE_DEVICE_TYPES` and tries encoder capture; records the failure. |
| `npu_capture_ops.py` | Op-level capture bisect (linear, conv1d, LSTM, SDPA, one_hot, index_add). |
| `multilingual_parity.py` | FLEURS en/de/ja/zh parity against a manifest TSV. |

Typical sequence on a fresh pod:

```bash
bash scripts/step0/restore_env.sh
export NEMOTRON_MODEL_PATH=/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b
export TBE_PARALLEL_COMPILER=0 TBE_OP_PARALLEL_COMPILER=0 MAX_COMPILE_CORE_NUMBER=1

# in-process checks (no server)
python scripts/step0/streaming_eager.py --dtype float32
python scripts/step0/bench_streaming.py --concurrency 1,2,4 --runs 3

# server checks
sgl-omni serve --model-path "$NEMOTRON_MODEL_PATH" --host 127.0.0.1 --port 8011 \
  --asr.factory.dtype float32 --asr.factory.enable_encoder_cuda_graph false --enable-realtime
python scripts/step0/realtime_client.py --port 8011
python scripts/step0/realtime_lifecycle.py --port 8011 --json-out logs/lifecycle_fp32.json
```

`realtime_lifecycle.py --audio-b` defaults to a 16 kHz mono PCM16 file; any clip
that the vendor/HF path transcribes as empty works, because the blank session
is there to prove state isolation, not accuracy (#4385).
