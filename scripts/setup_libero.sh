#!/bin/bash
# One-shot libero / libero_plus bootstrap for a new machine.
#
# - Locates the installed `libero` (hf-libero) and `libero_plus` packages
#   under $VENV_DIR (default: $PWD/.venv) — `find -L` because the venv may
#   be a symlink (e.g. Isambard scratch).
# - Writes ~/.libero/config.yaml and ~/.libero_plus/config.yaml so each
#   package's `__init__.py` skips its interactive `input()` prompt on
#   first import.
# - Downloads the LIBERO-plus assets bundle from HF into the libero_plus
#   site-packages dir if not already present, so AsyncVectorEnv workers
#   don't race to fetch them at first eval.
#
# Idempotent: safe to re-run.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$PWD/.venv}"
ASSETS_URL="${ASSETS_URL:-https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip}"

LIBERO_DIR=$(find -L "$VENV_DIR" -type d -path "*/libero/libero" -not -path "*libero_plus*" -print -quit)
LIBERO_PLUS_DIR=$(find -L "$VENV_DIR" -type d -path "*/libero_plus/libero_plus" -print -quit)

if [ -z "$LIBERO_PLUS_DIR" ]; then
  echo "libero_plus install not found under $VENV_DIR — run 'uv sync' first" >&2
  exit 1
fi

mkdir -p "$HOME/.libero_plus"
cat > "$HOME/.libero_plus/config.yaml" <<EOF
assets: $LIBERO_PLUS_DIR/assets
bddl_files: $LIBERO_PLUS_DIR/bddl_files
benchmark_root: $LIBERO_PLUS_DIR
datasets: $LIBERO_PLUS_DIR/../datasets
init_states: $LIBERO_PLUS_DIR/init_files
EOF

if [ -n "$LIBERO_DIR" ]; then
  mkdir -p "$HOME/.libero"
  cat > "$HOME/.libero/config.yaml" <<EOF
assets: $LIBERO_DIR/assets
bddl_files: $LIBERO_DIR/bddl_files
benchmark_root: $LIBERO_DIR
datasets: $LIBERO_DIR/../datasets
init_states: $LIBERO_DIR/init_files
EOF
fi

if [ ! -d "$LIBERO_PLUS_DIR/assets" ]; then
  TMP_EXTRACT=$(mktemp -d -p "$LIBERO_PLUS_DIR")
  curl -L -o "$TMP_EXTRACT/libero_assets.zip" "$ASSETS_URL"
  unzip -q "$TMP_EXTRACT/libero_assets.zip" -d "$TMP_EXTRACT"
  mv "$(find "$TMP_EXTRACT" -type d -name assets -print -quit)" "$LIBERO_PLUS_DIR/assets"
  rm -rf "$TMP_EXTRACT"
fi

echo "libero_plus ready at $LIBERO_PLUS_DIR"
[ -n "$LIBERO_DIR" ] && echo "libero ready at $LIBERO_DIR"
