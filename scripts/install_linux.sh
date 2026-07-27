#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Installing Linux system packages for GhostGUI..."
sudo apt update
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    libgl1 \
    libegl1 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-xinput0

if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
else
    echo "Reusing existing .venv."
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

echo
echo "GhostGUI installed successfully."
echo "Run it with:"
echo "  source .venv/bin/activate"
echo "  ghostgui"
echo
echo "Or use:"
echo "  bash scripts/run_linux.sh"
