#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting ICASSP Stuttering Audio Degradation Pipeline"
echo "=========================================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo "[1/3] Running Day-0 Gate pass to verify thesis decision rule..."
python3 icassp/day0_gate.py

echo "[2/3] Running full experimental pipeline (Layer selection, Exp A, Exp B, Exp C)..."
python3 icassp/main.py

echo "=========================================================="
echo " Pipeline Finished Successfully!"
echo " All results and vector figures saved to icassp/results/"
echo "=========================================================="
