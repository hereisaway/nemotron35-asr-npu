#!/usr/bin/env bash
# Restore the Nemotron 3.5 ASR on Ascend NPU Step 0 environment.
#
# Validated on: 1x Ascend 910B4 (A2, 32 GB HBM), driver 25.5.1, CANN 9.2.1,
# Python 3.11.18, torch 2.9.0+cpu / torch_npu 2.9.0, sglang 0.5.23 from
# pyproject_npu.toml, transformers 5.19.0, triton-ascend 3.2.0.
#
# Usage:
#   bash scripts/step0/restore_env.sh [--workdir /opt/atomgit] [--skip-model]
#
# The script is idempotent: existing checkouts and model files are reused.
set -euo pipefail

WORKDIR=${WORKDIR:-/opt/atomgit}
VENV=${VENV:-/opt/venv}
FORK_REPO=${FORK_REPO:-https://github.com/hereisaway/sglang-omni.git}
FORK_BRANCH=${FORK_BRANCH:-feat/nemotron35-asr-npu}
SGLANG_REF=${SGLANG_REF:-v0.5.23}
HF_MODEL=${HF_MODEL:-nvidia/nemotron-3.5-asr-streaming-0.6b}
SKIP_MODEL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) WORKDIR=$2; shift 2 ;;
    --skip-model) SKIP_MODEL=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

MODEL_DIR="$WORKDIR/models/nemotron-3.5-asr-streaming-0.6b"
OMNI_DIR="$WORKDIR/sglang-omni"
SGLANG_SRC="$WORKDIR/sglang"

export TBE_PARALLEL_COMPILER=0
export TBE_OP_PARALLEL_COMPILER=0
export MAX_COMPILE_CORE_NUMBER=1

log() { printf '\n==> %s\n' "$*"; }

log "checking NPU and toolchain"
npu-smi info >/dev/null
python -c 'import torch, torch_npu; assert torch.npu.is_available()' \
  || { echo "torch_npu cannot see an NPU; fix the pod first" >&2; exit 1; }

if [[ ! -w "$VENV" ]]; then
  log "taking ownership of $VENV"
  sudo chown -R "$(id -un):$(id -gn)" "$VENV"
fi

log "huggingface_hub / hf_transfer"
pip install -U "huggingface_hub[hf_xet]" hf_transfer >/dev/null

log "sglang $SGLANG_REF from source (pyproject_npu.toml)"
if [[ ! -d "$SGLANG_SRC/.git" ]]; then
  git clone --depth 1 --branch "$SGLANG_REF" \
    https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
fi
cp "$SGLANG_SRC/python/pyproject_npu.toml" "$SGLANG_SRC/python/pyproject.toml"
( cd "$SGLANG_SRC" && SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_REF#v}" \
    pip install -e python )

# pyproject_npu.toml does not pin torch, so the resolver may pull a newer CUDA
# torch. Re-pin the validated NPU stack afterwards.
log "re-pin torch / torch_npu to the validated stack"
pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" \
  "numpy==1.26.4"
pip install "torch_npu==2.9.0"

log "sglang-omni fork branch $FORK_BRANCH"

# The branch head as a tarball. Also repairs a clone whose checkout stopped
# part-way: git can exit 0 while leaving the worktree incomplete on flaky
# links, and extracting over it keeps the .git directory and the same tree.
fetch_omni_tarball() {
  echo "falling back to the codeload tarball of $FORK_BRANCH" >&2
  mkdir -p "$OMNI_DIR"
  curl -L --retry 5 --retry-all-errors \
    "https://codeload.github.com/hereisaway/sglang-omni/tar.gz/refs/heads/$FORK_BRANCH" \
    | tar xz -C "$OMNI_DIR" --strip-components=1
}

if [[ ! -d "$OMNI_DIR/.git" ]]; then
  git -c http.version=HTTP/1.1 clone --depth 1 --single-branch \
    --branch "$FORK_BRANCH" "$FORK_REPO" "$OMNI_DIR" \
    || fetch_omni_tarball
fi

# The package directory is the cheapest proof that the checkout is usable;
# a partial clone otherwise fails later at `pip install -e .` with a
# misleading ModuleNotFoundError.
if [[ ! -f "$OMNI_DIR/sglang_omni/__init__.py" ]]; then
  fetch_omni_tarball
fi
cd "$OMNI_DIR"

log "sglang-omni dependencies (pyproject_npu.toml, base stack pinned)"
python - <<'PY' > /tmp/omnireq.txt
import tomllib
d = tomllib.load(open("pyproject_npu.toml", "rb"))
print("\n".join(d["project"]["dependencies"]))
PY
printf '%s\n' \
  'torch==2.9.0' 'torch_npu==2.9.0' \
  'numpy==1.26.4' 'pandas==2.2.3' 'soundfile==0.13.1' \
  'torchaudio==2.9.0' 'torchvision==0.24.0' > /tmp/omni_constraints.txt
pip install -r /tmp/omnireq.txt -c /tmp/omni_constraints.txt

# sglang pulls upstream triton (3.9.x); triton-ascend ships its own `triton`
# package. Uninstall both and install triton-ascend alone, otherwise stale
# NVIDIA backend files break `import triton` (`cannot import name 'Language'`).
pip uninstall -y triton triton-ascend >/dev/null 2>&1 || true
rm -rf "$VENV/lib/python3.11/site-packages/triton"
pip install "triton-ascend==3.2.0" pybind11 "torchaudio==2.9.0" \
  pytest pytest-asyncio ruff jiwer -c /tmp/omni_constraints.txt
pip install -e . --no-deps

if [[ "$SKIP_MODEL" -eq 0 ]]; then
  log "model weights"
  mkdir -p "$MODEL_DIR"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    hf download "$HF_MODEL" --local-dir "$MODEL_DIR" \
      config.json generation_config.json processor_config.json \
      tokenizer.json tokenizer_config.json model.safetensors
fi

log "verification"
python - <<'PY'
import torch, torch_npu, transformers
assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torch.npu.is_available()
print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__,
      "transformers:", transformers.__version__)
PY
python -c "import sglang; print('sglang:', sglang.__version__)"
python -c "import sglang_omni; print('sglang-omni ok')"
( cd "$OMNI_DIR" && python -m pytest tests/unit_test/nemotron3_5_asr -q )

log "done"
echo "export NEMOTRON_MODEL_PATH=$MODEL_DIR"
