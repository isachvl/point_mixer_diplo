#!/bin/sh
echo "[PM INFO] Installing cuda operations..."

python3 - <<'PY'
import importlib
import torch

try:
    importlib.import_module("pointops2_cuda")
except Exception:
    raise SystemExit(1)

print("[PM INFO] pointops2_cuda already installed. Skip build.")
PY
if [ "$?" -eq 0 ]; then
    exit 0
fi

cd ./lib/pointops2
python3 setup.py install
cd ../../..

echo "[PM INFO] Done !"
