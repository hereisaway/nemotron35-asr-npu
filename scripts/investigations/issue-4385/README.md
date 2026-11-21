# #4385 probe scripts

Scripts used to root-cause the blank Nemotron 3.5 ASR transcripts and to
re-measure parity after the fix. Findings:
[docs/issue-4385-investigation-2026-11-21.md](../../../docs/issue-4385-investigation-2026-11-21.md).

Paths below assume the pod layout used on 2026-11-21 (`/opt/atomgit`). Change
`MODEL_DIR` and `NEMO_PATH` if the checkpoints live elsewhere.

## Reference environment

The NeMo reference runs in its own venv so the validated NPU stack in
`/opt/venv` stays untouched:

```bash
python3 -m venv /opt/atomgit/venv-nemo
echo "/opt/venv/lib/python3.11/site-packages" \
  > /opt/atomgit/venv-nemo/lib/python3.11/site-packages/omni_stack.pth
printf '%s\n' 'torch==2.9.0' 'torchvision==0.24.0' 'torchaudio==2.9.0' 'numpy==1.26.4' \
  > /tmp/nemo_constraints.txt
/opt/atomgit/venv-nemo/bin/pip install "nemo_toolkit[asr]" -c /tmp/nemo_constraints.txt
# pyarrow/datasets pins that keep numpy at 1.26.4
/opt/atomgit/venv-nemo/bin/pip install "pyarrow==16.1.0" "numba==0.60.0" "datasets==2.19.0"
```

The `.pth` makes the venv reuse the validated torch build; packages installed
into the venv shadow it. Checkpoints (Hugging Face is not reachable directly
from the pod, use the mirror):

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  nvidia/nemotron-3.5-asr-streaming-0.6b \
  --local-dir /opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b \
  config.json generation_config.json processor_config.json tokenizer.json \
  tokenizer_config.json model.safetensors
HF_ENDPOINT=https://hf-mirror.com hf download \
  nvidia/nemotron-3.5-asr-streaming-0.6b nemotron-3.5-asr-streaming-0.6b.nemo \
  --local-dir /opt/atomgit/models/nemo
```

## Data

```bash
bash prepare_fleurs.sh        # 4 locales x 4 clips -> /tmp/fleurs_audio/manifest.tsv
```

## Run order

1. `nemo_reference.py` — NeMo transcripts plus encoder/feature dumps under
   `/tmp/nemo_*.npy`. Run with `/opt/atomgit/venv-nemo/bin/python`.
2. `compare_nemo_hf.py`, `compare_shift.py`, `compare_subsampling.py`,
   `compare_pos_layer0.py`, `compare_layer_blocks.py` — stage-by-stage diffs
   against the HF port (main venv).
3. `check_attention.py`, `check_sdpa.py` — attention output and mask inspection.
4. `test_fix.py`, `test_position_bias_fix.py` — default vs fixed mask handling.
5. `multilingual_parity_cpu.py` — fixed HF path parity on CPU; combine with
   `scripts/step0/multilingual_parity.py` for the vendored NPU path.

`probe_*.py` and `repro_hf.py` are the earlier elimination probes (frontend,
normalization, prompt, length, onset, VAD, alignment, per-layer norms).

## Key result

`sdpa_attention_forward` converts any non-bool attention mask to boolean when
`torch_npu` is importable. The model passes its relative-position bias as that
mask, so every position is masked and self-attention returns zeros. Fix:
[PR #4445](https://github.com/sgl-project/sglang-omni/pull/4445).
