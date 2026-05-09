#!/bin/bash
set -euo pipefail

sudo apt-get update && sudo apt-get install -y \
  cmake \
  ffmpeg \
  g++ \
  libegl1 \
  libexpat1 \
  libfontconfig1-dev \
  libgl1 \
  libglvnd0 \
  libmagickwand-dev \
  libopengl0 \
  unzip

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/sky_workdir

uv sync

bash ~/sky_workdir/scripts/setup_libero.sh

cd ~/sky_workdir
uv run python ~/sky_workdir/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
