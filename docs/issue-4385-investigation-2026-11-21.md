# #4385 investigation — 2026-11-21

Follow-up probes on `nvidia/nemotron-3.5-asr-streaming-0.6b` after filing #4385.
Environment: 1x Ascend 910B4 (A2), CANN 9.2.1, torch 2.9.0+cpu, torch_npu 2.9.0,
transformers 5.19.0, sglang-omni `81c68c20`. CPU probes used the same
transformers build; the NPU probe used the vendored copy in
`sglang_omni/vendor/nemotron3_5_asr`.

## Reproduction (confirmed on CPU and NPU)

| Clip | HF CPU fp32 | sglang-omni vendored NPU fp32 |
|---|---|---|
| `bcn_weather.mp3` (11.0 s, card example) | 140/140 blank, `''` | 140/140 blank, `''` |
| `tests/data/query_to_draw.wav` (5.06 s, 16 kHz) | `What is happening in this videos are feeling?` | `What is happening in this videos are.` |
| `tests/data/query_to_cars.wav` (4.62 s, 48 kHz) | 60/60 blank, `''` | 60/60 blank, `''` |

The HF and vendored paths agree, so this is a shared model-implementation bug,
not a platform or sglang-omni streaming-registry bug.

## Ruled out

1. **Mel frontend.** A line-by-line NeMo-style reference (preemphasis 0.97,
   `hann_window(periodic=False)`, `n_fft=512`, `hop=160`, librosa slaney mel,
   `log(x + 2**-24)`) matches the extractor output exactly: max abs difference
   `0.0000` on all three clips. The extractor's own comment says the model never
   normalizes the mel features; feeding per-feature or global normalized features
   makes the working clip blank as well, so the checkpoint expects the raw
   log-mel scale.
2. **Prompt / language.** `en-US` resolves to prompt id 0, `auto` to 101;
   both blank on the failing clips, so the prompt is not the trigger.
3. **Decoder loop.** Traced `decoder.forward`: every step receives a single
   token, the LSTM cache is only updated on non-blank emissions, and the
   all-blank fast path returns the frozen cache. Behaviour matches RNN-T greedy
   decoding; it is not the source of the blanks.
4. **Checkpoint loading.** `output_loading_info=True` reports 0 missing,
   0 unexpected, 0 mismatched keys.
5. **Audio preparation.** Removing DC, fading in, peak-normalizing, trimming
   lead/tail silence (energy VAD at 30/40 dB) and gain scaling by 0.25x/4x do not
   produce a transcript for the failing clips. Length is not the trigger either:
   the 3 s prefix of `bcn_weather` is already all-blank.

## Root cause: the torch_npu SDPA mask conversion in Transformers

The frontend, the checkpoint and the encoder all match the reference up to the
first self-attention call:

| Stage | NeMo vs HF cosine |
|---|---|
| mel features | max abs diff 1.4e-4 (dither 1e-5) |
| subsampling | 1.0000 (max abs diff 2e-3) |
| positional encoding | 1.0000 |
| layer-0 feed-forward 1 | 1.0000 |
| **layer-0 self-attention** | **HF output is exactly 0** |
| layer-0 convolution | 0.86 |
| full encoder | 0.70 median |

`NemotronAsrStreamingEncoderAttention` passes its relative-position bias matrix
`matrix_bd` (float, `-inf` for disallowed pairs) as `attention_mask` to the
attention interface. Transformers 5.19 `sdpa_attention_forward` has an
NPU-specific branch:

```python
if _is_torch_npu_available:
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)
```

That branch assumes an additive mask whose zeros mean "keep". The bias matrix
has no exact zeros, so `bool()` makes every entry `True` and `logical_not`
makes every entry `False`, which SDPA reads as "every position is masked". The
attention output is then exactly zero on every frame, and the encoder falls
back to its convolution/feed-forward residual path: partially intelligible for
easy audio, all-blank for the rest.

The branch is gated on **`torch_npu` being importable**, not on the tensor
device, so it also fires on CPU in any environment where `torch_npu` is
installed. That is why the "HF CPU" runs blanked on the NPU pod.

Disabling the branch for the same three clips reproduces the NeMo reference:

| Clip | Default (torch_npu conversion) | Flag disabled | NeMo reference |
|---|---|---|---|
| `bcn_weather` | `''` | `Yesterday it was uh thirty five degrees in Barcelona, but today the temperature will go down to minus uh twenty degrees.` | same |
| `query_to_draw` | `What is happening in this videos are feeling?` | `What is happening in this video?  Answer in ten words or fewer.` | same |
| `query_to_cars` | `''` | `Homening cars are there in the picture.` | same |

Setting `_attn_implementation="eager"` is not a workaround: it returns
degenerate output (590 tokens / 1 blank on `query_to_cars`).

### Fix options

1. Transformers: only apply the NPU conversion to masks that follow the
   `0`/`-inf` additive convention, or skip it when the caller passes a
   `position_bias`. The interface already has a `position_bias` argument.
2. Model port: pass the relative-position bias through `position_bias=` instead
   of `attention_mask`, and reserve `attention_mask` for padding.
3. Local workaround on an NPU pod, until either lands:
   `transformers.integrations.sdpa_attention._is_torch_npu_available = False`
   (validated above; it also disables the NPU optimization for every model in
   the process).

## Separate bug found: blank/pad token ids disagree

| Source | `<blank>` | `<pad>` |
|---|---|---|
| `config.json` | 13087 (`blank_token_id`), `pad_token_id` = 0 | - |
| tokenizer | **13088** (out of range, `vocab_size` = 13088) | **13087** |

`processor.decode()` still strips blank frames because `skip_ids` includes the
tokenizer's pad id (13087), which happens to equal the model's blank id. But
`processor(text=...)` prepends `self.blank_token_id` = 13088 to build
`decoder_input_ids`, which is outside the embedding table and would raise on any
label-encoding path. This is independent of the blank-transcript bug and easy to
fix upstream.

## Reference setup

`nemo_toolkit==3.0.0` in an isolated venv (`/opt/atomgit/venv-nemo`, reuses the
validated torch 2.9.0 through a `.pth`), with the official
`nemotron-3.5-asr-streaming-0.6b.nemo` checkpoint. `pyarrow 16.1.0` and
`datasets 2.19.0` are required to keep numpy at 1.26.4. The NeMo config confirms
`normalize: NA`, `dither: 1e-5`, `pad_to: 0`, `causal_downsampling: true`,
`chunked_limited` with `[-56, 3]`, which matches the HF port's frontend.

## Reproduction scripts

`/opt/atomgit/audio/` (this pod):

- `repro_hf.py` — token/blank statistics for one clip.
- `probe_norm.py` — raw vs per-feature vs global normalization.
- `probe_encoder.py` — encoder output and attention-mask variants.
- `probe_logits.py` — checkpoint load info and joint logits.
- `probe_length.py`, `probe_lookahead.py`, `probe_context.py` — length, lookahead
  and attention-window sweeps.
- `probe_onset.py`, `probe_vad.py`, `probe_align.py` — onset, trimming and
  alignment probes.
- `probe_nemo_features.py` — reference frontend comparison.
- `probe_layers.py` — per-layer encoder norm trace.
- `nemo_reference.py` — NeMo transcripts plus encoder/feature dumps.
- `compare_nemo_hf.py`, `compare_shift.py`, `compare_subsampling.py`,
  `compare_pos_layer0.py`, `compare_layer_blocks.py` — stage-by-stage diffs.
- `check_attention.py`, `check_sdpa.py` — mask and attention output inspection.
- `test_fix.py` — the default vs flag-disabled vs eager comparison above.
