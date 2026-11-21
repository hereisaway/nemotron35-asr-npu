"""Bisect NPU graph capture: one target per process (a failed capture poisons the process).

Usage:
  PER_STREAM_QUEUE=1 TASK_QUEUE_ENABLE=1 python npu_capture_encoder.py --target encoder
Targets: encoder, subsampling, positions, projector, joint, block, ff1, attn, conv
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PER_STREAM_QUEUE", "1")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.pop("ASCEND_LAUNCH_BLOCKING", None)

import torch  # noqa: E402
import torch_npu  # noqa: E402

from sglang_omni.vendor.nemotron3_5_asr.configuration_nemotron3_5_asr import (  # noqa: E402
    Nemotron3_5AsrConfig,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (  # noqa: E402
    Nemotron3_5AsrForRNNT,
)

MODEL = os.environ["NEMOTRON_MODEL_PATH"]
DEVICE = torch.device("npu")


def build_target(model, name: str):
    encoder_config = model.config.encoder_config
    mel_bins = int(encoder_config.num_mel_bins)
    hidden = int(encoder_config.hidden_size)
    features = torch.randn(1, 25, mel_bins, device=DEVICE)
    if name == "encoder":
        return lambda: model.encoder(features, attention_mask=None, use_cache=False)
    if name == "subsampling":
        return lambda: model.encoder.subsampling(features, attention_mask=None)
    if name == "positions":
        hidden_states = torch.randn(1, 8, hidden, device=DEVICE)
        return lambda: model.encoder.encode_positions(hidden_states, cached_frames=0)
    if name == "projector":
        return lambda: model.encoder_projector(
            torch.randn(1, 8, hidden, device=DEVICE)
        )
    if name == "block":
        hidden_states = torch.randn(1, 8, hidden, device=DEVICE)
        positions = model.encoder.encode_positions(hidden_states, cached_frames=0)
        layer = model.encoder.layers[0]
        return lambda: layer(
            hidden_states,
            attention_mask=None,
            all_masked_rows=None,
            position_embeddings=positions,
            past_key_values=None,
            padding_cache=None,
            use_cache=False,
        )
    if name == "ff1":
        hidden_states = torch.randn(1, 8, hidden, device=DEVICE)
        return lambda: model.encoder.layers[0].feed_forward1(hidden_states)
    if name == "attn":
        hidden_states = torch.randn(1, 8, hidden, device=DEVICE)
        positions = model.encoder.encode_positions(hidden_states, cached_frames=0)
        return lambda: model.encoder.layers[0].self_attn(
            hidden_states, position_embeddings=positions
        )
    if name == "conv":
        hidden_states = torch.randn(1, 8, hidden, device=DEVICE)
        return lambda: model.encoder.layers[0].conv(hidden_states)
    if name == "joint":
        decoder_hidden = int(model.config.decoder_hidden_size)
        return lambda: model.joint(
            torch.randn(1, decoder_hidden, device=DEVICE),
            torch.randn(1, decoder_hidden, device=DEVICE),
        )
    raise SystemExit(f"unknown target: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--attn-impl",
        default="",
        help="override the encoder attention implementation, e.g. eager or sdpa",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    torch.npu.set_device(DEVICE)
    torch.npu.config.allow_internal_format = False
    config = Nemotron3_5AsrConfig.from_pretrained(MODEL, local_files_only=True)
    if args.attn_impl:
        config._attn_implementation = args.attn_impl
        config.encoder_config._attn_implementation = args.attn_impl
    model = Nemotron3_5AsrForRNNT.from_pretrained(
        MODEL, config=config, dtype=torch.float32, local_files_only=True
    ).to(DEVICE)
    model.eval()
    build = build_target(model, args.target)

    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(npu_graph=graph):
            build()
        torch.npu.synchronize()
        result = {
            "target": args.target,
            "attn_impl": args.attn_impl or "default",
            "captured": True,
        }
    except Exception as error:  # noqa: BLE001
        result = {
            "target": args.target,
            "attn_impl": args.attn_impl or "default",
            "captured": False,
            "error_type": type(error).__name__,
            "error": str(error).splitlines()[0][:500],
        }
    payload = json.dumps(result)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
