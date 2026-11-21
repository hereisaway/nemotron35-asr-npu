"""Fixed HF-path FLEURS parity on CPU (torch_npu mask conversion disabled)."""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import jiwer
import librosa
import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.integrations import sdpa_attention

MODEL_DIR = "/opt/atomgit/models/nemotron-3.5-asr-streaming-0.6b"
CJK_LANGS = {"ja_jp", "cmn_hans_cn"}


def normalize(text: str, lang: str) -> str:
    text = text.lower().strip()
    if lang in CJK_LANGS:
        return "".join(re.sub(r"\s+", "", text))
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            lang, audio, reference = line.rstrip("\n").split("\t")
            rows.append({"lang": lang, "audio": audio, "reference": reference})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/tmp/fleurs_audio/manifest.tsv")
    parser.add_argument("--output", default="/tmp/parity_fixed_cpu_fp32.json")
    parser.add_argument("--normalize", choices=["none", "peak", "rms"], default="peak")
    args = parser.parse_args()

    sdpa_attention._is_torch_npu_available = False
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    results = []
    for row in load_manifest(args.manifest):
        waveform, sample_rate = librosa.load(row["audio"], sr=16000)
        if args.normalize == "peak":
            peak = float(np.max(np.abs(waveform)))
            waveform = waveform * (0.9 / peak) if peak > 0 else waveform
        elif args.normalize == "rms":
            rms = float(np.sqrt((waveform**2).mean()))
            waveform = waveform * (0.06 / rms) if rms > 0 else waveform
        inputs = processor(waveform, sampling_rate=sample_rate, language="auto",
                           return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True)
        elapsed = time.perf_counter() - started
        text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        reference_norm = normalize(row["reference"], row["lang"])
        hypothesis_norm = normalize(text, row["lang"])
        wer = jiwer.wer(reference_norm, hypothesis_norm) if reference_norm else None
        results.append({**row, "transcript": text.strip(), "elapsed_s": round(elapsed, 3),
                        "wer": round(wer, 4) if wer is not None else None,
                        "ref_norm": reference_norm, "hyp_norm": hypothesis_norm})
        print(f"{row['lang']} {os.path.basename(row['audio'])} wer={wer:.4f} text={text.strip()!r}",
              flush=True)

    summary = {}
    for item in results:
        summary.setdefault(item["lang"], []).append(item["wer"])
    summary = {lang: round(sum(values) / len(values), 4) for lang, values in summary.items()}
    overall = round(sum(item["wer"] for item in results) / len(results), 4)
    summary["overall"] = overall
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"results": results, "summary": summary}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
