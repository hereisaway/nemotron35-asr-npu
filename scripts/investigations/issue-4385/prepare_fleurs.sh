#!/usr/bin/env bash
# Fetch four FLEURS test clips per locale and build the parity manifest.
set -euo pipefail

OUT=${OUT:-/tmp/fleurs_audio}
mkdir -p "$OUT"

for locale in en_us de_de ja_jp cmn_hans_cn; do
  if [[ ! -s "$OUT/$locale.tsv" ]]; then
    curl -sSL --retry 5 --retry-all-errors -m 120 \
      "https://hf-mirror.com/datasets/google/fleurs/resolve/main/data/$locale/test.tsv" \
      -o "$OUT/$locale.tsv"
  fi
  if [[ ! -s "$OUT/$locale.tar.gz" ]]; then
    echo "downloading $locale audio"
    curl -sSL --retry 5 --retry-all-errors -m 1800 \
      "https://hf-mirror.com/datasets/google/fleurs/resolve/main/data/$locale/audio/test.tar.gz" \
      -o "$OUT/$locale.tar.gz"
  fi
  echo "$locale tsv=$(wc -l < "$OUT/$locale.tsv") tar=$(du -h "$OUT/$locale.tar.gz" | cut -f1)"
done

python3 - <<'PY'
from pathlib import Path
import tarfile

root = Path("/tmp/fleurs_audio")
lines = []
for locale in ("en_us", "de_de", "ja_jp", "cmn_hans_cn"):
    rows = []
    with open(root / f"{locale}.tsv", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rows.append((parts[1], parts[2]))
    target = root / locale
    target.mkdir(exist_ok=True)
    with tarfile.open(root / f"{locale}.tar.gz") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith(".wav")]
        names = {Path(m.name).name: m for m in members}
        picked = 0
        for filename, reference in rows:
            member = names.get(filename)
            if member is None:
                continue
            archive.extract(member, path=target)
            path = target / member.name
            if not path.exists():
                path = target / Path(member.name)
            lines.append(f"{locale}\t{path}\t{reference}")
            picked += 1
            if picked == 4:
                break
    print(f"{locale}: {picked} clips, {len(members)} wavs in tar")

manifest = root / "manifest.tsv"
manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"manifest {manifest} rows={len(lines)}")
PY
