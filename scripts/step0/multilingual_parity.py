"""Run multilingual offline parity (fp32 vs bf16) on Ascend NPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time

import jiwer
import librosa
import numpy as np
import torch
import torch_npu  # noqa: F401
from sglang_omni.vendor.nemotron3_5_asr.configuration_nemotron3_5_asr import (
    Nemotron3_5AsrConfig,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)
from sglang_omni.vendor.nemotron3_5_asr.processing_nemotron3_5_asr import (
    Nemotron3_5AsrProcessor,
)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
CJK_LANGS = {"ja_jp", "cmn_hans_cn"}


def normalize(text: str, lang: str) -> str:
    text = text.lower().strip()
    if lang in CJK_LANGS:
        return " ".join(re.sub(r"\s+", "", text))
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_waveform(waveform, mode: str):
    if mode == "peak":
        peak = float(np.max(np.abs(waveform)))
        return waveform * (0.9 / peak) if peak > 0 else waveform
    if mode == "rms":
        rms = float(np.sqrt((waveform**2).mean()))
        return waveform * (0.06 / rms) if rms > 0 else waveform
    return waveform


def load_manifest(path: str, limit: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            lang, audio, reference = line.rstrip("\n").split("\t")
            rows.append({"lang": lang, "audio": audio, "reference": reference})
    if limit:
        rows = rows[:limit]
    return rows


def load_model(model_path: str, config, dtype):
    return (
        Nemotron3_5AsrForRNNT.from_pretrained(
            model_path, config=config, dtype=dtype, local_files_only=True
        )
        .to("npu")
        .eval()
    )


def run_dtype(model_path, processor, config, dtype_name, rows, normalize):
    dtype = DTYPES[dtype_name]
    model = load_model(model_path, config, dtype)
    results = []
    try:
        for row in rows:
            waveform, sample_rate = librosa.load(row["audio"], sr=16000)
            waveform = normalize_waveform(waveform, normalize)
            inputs = processor(
                [waveform],
                sampling_rate=sample_rate,
                language=["auto"],
                padding="longest",
                return_tensors="pt",
            ).to(device="npu", dtype=dtype)
            started_at_s = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, return_dict_in_generate=True)
            elapsed_s = time.perf_counter() - started_at_s
            text = processor.batch_decode(
                output.sequences.detach().to("cpu"), skip_special_tokens=True
            )[0]
            results.append(
                {
                    "lang": row["lang"],
                    "audio": os.path.basename(row["audio"]),
                    "reference": row["reference"],
                    "transcript": text.strip(),
                    "elapsed_s": round(elapsed_s, 3),
                }
            )
            print(
                f"{dtype_name} {row['lang']} {os.path.basename(row['audio'])} "
                f"elapsed_s={elapsed_s:.3f} text={text.strip()!r}",
                flush=True,
            )
    finally:
        del model
        gc.collect()
        torch.npu.empty_cache()
    return results


def add_wer(results):
    for item in results:
        lang = item["lang"]
        ref = normalize(item["reference"], lang)
        hyp = normalize(item["transcript"], lang)
        item["wer"] = round(jiwer.wer(ref, hyp), 4) if ref else None
        item["ref_norm"] = ref
        item["hyp_norm"] = hyp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("NEMOTRON_MODEL_PATH"),
        required=os.environ.get("NEMOTRON_MODEL_PATH") is None,
    )
    parser.add_argument("--manifest", default="/tmp/fleurs_audio/manifest.tsv")
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--normalize", choices=["none", "peak", "rms"], default="peak")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = load_manifest(args.manifest, args.limit)
    processor = Nemotron3_5AsrProcessor.from_pretrained(
        args.model_path, local_files_only=True
    )
    config = Nemotron3_5AsrConfig.from_pretrained(args.model_path)

    by_dtype = {}
    for dtype_name in args.dtypes.split(","):
        by_dtype[dtype_name] = run_dtype(
            args.model_path, processor, config, dtype_name, rows, args.normalize
        )
        add_wer(by_dtype[dtype_name])

    pairwise = None
    if "float32" in by_dtype and "bfloat16" in by_dtype:
        pairwise = []
        for fp32, bf16 in zip(by_dtype["float32"], by_dtype["bfloat16"]):
            lang = fp32["lang"]
            pairwise.append(
                {
                    "lang": lang,
                    "audio": fp32["audio"],
                    "exact_match": fp32["transcript"] == bf16["transcript"],
                    "wer_fp32_vs_bf16": round(
                        jiwer.wer(fp32["hyp_norm"], bf16["hyp_norm"]), 4
                    ),
                }
            )

    summary = {}
    for dtype_name, results in by_dtype.items():
        by_lang = {}
        for item in results:
            by_lang.setdefault(item["lang"], []).append(item["wer"])
        summary[dtype_name] = {
            lang: round(sum(wers) / len(wers), 4) for lang, wers in by_lang.items()
        }

    payload = {
        "model_path": args.model_path,
        "normalize": args.normalize,
        "by_dtype": by_dtype,
        "pairwise_fp32_vs_bf16": pairwise,
        "mean_wer": summary,
    }
    output = args.output or f"logs/multilingual_parity_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"wrote {output}")
    print(json.dumps(summary, ensure_ascii=False))
    if pairwise:
        matches = sum(1 for item in pairwise if item["exact_match"])
        print(f"fp32/bf16 exact transcript matches: {matches}/{len(pairwise)}")
        print(json.dumps(pairwise, ensure_ascii=False))


if __name__ == "__main__":
    main()
