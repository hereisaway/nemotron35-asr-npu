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

## Localized

- The encoder/joint numerics diverge between clips. On the failing clips the
  encoder output spikes to a frame norm of ~20 immediately after the onset
  (`bcn_weather` 20.5, `query_to_cars` 21.0) while the working clip stays at
  4.25; joint logits then reach -920 and +3500 with blank winning every frame.
  On the working clip logits stay within a few tens and non-blank tokens win
  occasionally.
- **Onset sensitivity is the strongest lead.** Removing the first 0.25 s of
  `query_to_draw` yields the exact reference transcript
  (`What is happening in this video`), while adding 0.5 s of leading silence
  degrades it into a partial/garbled transcript. That points at the encoder's
  cache-aware streaming prefix handling (`init_pad` / `left_pad` in
  `NemotronAsrStreamingConv1dPaddingCache` and the first-chunk path), not at the
  weights. The GGUF reference transcribing the same clips proves the checkpoint
  and the audio are fine.
- `num_lookahead_tokens` (0/3/6/13/57) and `encoder.config.sliding_window`
  (57 → 5000) produce byte-identical tokens in the offline `generate()` path.
  Either those knobs do not reach the encoder in the offline path, or the
  attention mask is fixed at construction; worth checking separately.

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

## Suggested next step

Dump encoder frames from a NeMo runtime for the same clip (the NeMo-Speech.cpp
GGUF build used in Step 0, or `nemo_toolkit`) and diff them against
`model.encoder(...)`. The frontend already matches, so the first divergent frame
identifies the layer; the onset experiment says to look at the first chunk /
`init_pad` handling.

## Reproduction scripts

`/opt/atomgit/audio/probe_*.py` (this pod):

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
