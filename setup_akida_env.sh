#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
VENV_DIR=".ai4fdir"
REQ_FILE="requirements.txt"

echo "==> Creating Python venv at: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing TensorFlow 2.15.0.post1 with CUDA libraries via pip"
python -m pip install "tensorflow[and-cuda]==2.15.0.post1"

echo "==> Installing OpenCV (as in def file)"
python -m pip install --prefer-binary "opencv-python==4.11.0.86"

# Optional project requirements
if [[ -f "${REQ_FILE}" ]]; then
  echo "==> Installing project requirements from ${REQ_FILE}"
  python -m pip install --prefer-binary -r "${REQ_FILE}"
else
  echo "==> No ${REQ_FILE} found; skipping project requirements."
fi

echo "==> Installing MetaTF / Akida toolchain (v1)"
python -m pip install --prefer-binary \
  "akida==2.16.1" \
  "cnn2snn==2.16.1" \
  "akida-models==1.10.0" \
  "quantizeml==0.19.0"

# --- VS Code integration ---
echo "==> Writing VS Code settings and .env"

mkdir -p .vscode
cat > .vscode/settings.json << 'JSON'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.envFile": "${workspaceFolder}/.env"
}
JSON

# VS Code reads environment variables from .env 
cat > .env << 'ENV'
# CUDA paths (usually not needed when using tensorflow[and-cuda], but kept for parity)
CUDA_HOME=/usr/local/cuda
CUDA_ROOT=${CUDA_HOME}
PATH=${PATH}:${CUDA_ROOT}/bin:/usr/local/bin
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_ROOT}/lib64

# TensorFlow & Python runtime noise/perf
TF_CPP_MIN_LOG_LEVEL=2
OMP_NUM_THREADS=1
PYTHONUNBUFFERED=1
ENV

# --- test script  ---
cat > verify_env.py << 'PY'
import tensorflow as tf
import importlib

def safe_ver(pkg):
    try:
        m = importlib.import_module(pkg)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"not installed ({e})"

print("TF:", tf.__version__)
print("akida:", safe_ver("akida"))
print("cnn2snn:", safe_ver("cnn2snn"))
print("quantizeml:", safe_ver("quantizeml"))
PY

echo "==> Verifying versions…"
python verify_env.py

echo "==> Done!"
echo "Open VS Code here (code .). It should auto-select: ${VENV_DIR}/bin/python"
