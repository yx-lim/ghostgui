#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

HOST_ARCH="$(uname -m)"
NATIVE_ARCH="$HOST_ARCH"
if [ "$(sysctl -in hw.optional.arm64 2>/dev/null || true)" = "1" ]; then
    NATIVE_ARCH="arm64"
fi

PYTHON_BIN=""

case "$NATIVE_ARCH" in
arm64)
    if [ -x "/opt/homebrew/bin/python3" ]; then
        PYTHON_BIN="/opt/homebrew/bin/python3"
    elif [ -x "/opt/homebrew/bin/brew" ]; then
        echo "Installing arm64 Python with /opt/homebrew/bin/brew..."
        /opt/homebrew/bin/brew install python
        PYTHON_BIN="/opt/homebrew/bin/python3"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    fi
    ;;
x86_64)
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    fi
    ;;
*)
    echo "Unsupported macOS architecture: $NATIVE_ARCH"
    exit 1
    ;;
esac

if [ -z "$PYTHON_BIN" ]; then
    echo "python3 was not found."
    if command -v brew >/dev/null 2>&1; then
        echo "Homebrew is available. Install Python with:"
        echo "  brew install python"
    else
        echo "Install Python 3.10 or newer from https://www.python.org/ or Homebrew."
    fi
    exit 1
fi

PYTHON_ARCH="$("$PYTHON_BIN" -c 'import platform; print(platform.machine())')"

if [ "$PYTHON_ARCH" != "$NATIVE_ARCH" ]; then
    echo "This Mac needs a native $NATIVE_ARCH Python, but $PYTHON_BIN is $PYTHON_ARCH."
    if [ "$NATIVE_ARCH" = "arm64" ]; then
        echo
        echo "On Apple Silicon, install an arm64 Python and rerun this script."
        echo "Recommended Homebrew path:"
        echo "  /opt/homebrew/bin/brew install python"
        echo
        echo "If Homebrew is installed under /usr/local, it is probably the Intel/Rosetta build."
    else
        echo
        echo "On Intel Mac, install an x86_64 Python and rerun this script."
    fi
    exit 1
fi

if command -v brew >/dev/null 2>&1; then
    echo "Homebrew found."
fi

if [ "$HOST_ARCH" != "$NATIVE_ARCH" ]; then
    echo "Native Mac architecture is $NATIVE_ARCH; current shell reports $HOST_ARCH."
    echo "Continuing with native $NATIVE_ARCH Python."
fi
echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version), $PYTHON_ARCH)"

if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "Reusing existing .venv."
fi

source .venv/bin/activate

VENV_ARCH="$(python -c 'import platform; print(platform.machine())')"
if [ "$VENV_ARCH" != "$NATIVE_ARCH" ]; then
    echo ".venv is using a $VENV_ARCH Python, but this Mac requires $NATIVE_ARCH."
    echo "Remove the existing .venv and rerun this script:"
    echo "  rm -rf .venv"
    echo "  bash scripts/install_macos.sh"
    exit 1
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

if ! command -v mjpython >/dev/null 2>&1; then
    echo "mjpython was not found after installing MuJoCo."
    echo "The main GhostGUI app can still launch, but the standalone MuJoCo passive"
    echo "viewer requires mjpython on macOS."
    exit 1
fi

MJPYTHON_ARCH="$(mjpython -c 'import platform; print(platform.machine())')"
if [ "$MJPYTHON_ARCH" != "$NATIVE_ARCH" ]; then
    echo "mjpython is $MJPYTHON_ARCH, but this Mac requires $NATIVE_ARCH."
    echo "Remove .venv and recreate it with a native Python."
    exit 1
fi

echo
echo "GhostGUI installed successfully."
echo "Run it with:"
echo "  source .venv/bin/activate"
echo "  ghostgui"
echo
echo "MuJoCo passive viewer launcher:"
echo "  $(command -v mjpython)"
echo
echo "Or use:"
echo "  bash scripts/run_macos.sh"
