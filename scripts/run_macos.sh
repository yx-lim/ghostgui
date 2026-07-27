#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -f ".venv/bin/activate" ]; then
    echo ".venv was not found. Run bash scripts/install_macos.sh first."
    exit 1
fi

source .venv/bin/activate
ghostgui "$@"
