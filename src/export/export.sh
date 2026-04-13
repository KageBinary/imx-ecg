#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Path to the trained PyTorch checkpoint (.pth or .pt file).
# This is the saved model weights from training — the export converts these
# into formats that can run without PyTorch installed.
CHECKPOINT="${1:-outputs/models/best_model.pth}"

# The Python class name of the model architecture (e.g. ECGCNN, ECGFFTGlobalPoolNet).
# Must match the class used during training so the weights load correctly.
MODEL_CLASS="${2:-ECGCNN}"

# The Python module where the model class is defined, relative to src/.
# "ecgcnn.model" = src/ecgcnn/model.py, "fft_gp.models_fft_gp" = src/fft_gp/models_fft_gp.py, etc.
MODULE="${3:-ecgcnn.model}"

# Number of ECG samples the model expects as input.
# 10000 = the fixed signal length ECGCNN was trained on (from config.py).
# Must match what was used during training — signals get padded/truncated to this.
INPUT_LENGTH="${4:-10000}"

# Number of classification categories the model outputs.
# 4 = Normal (N), AFib (A), Other (O), Noisy (~) — the PhysioNet 2017 challenge classes.
NUM_CLASSES="${5:-4}"

# Directory where exported model files are saved.
OUTPUT_DIR="exports"

ONNX_OUT="${OUTPUT_DIR}/ecg_model.onnx"
TFLITE_OUT="${OUTPUT_DIR}/ecg_model.tflite"

# --- Check checkpoint exists ---
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Usage: ./export.sh <checkpoint> [model_class] [module] [input_length] [num_classes]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Step 1/3: Export to ONNX ==="
python src/export/export_onnx.py \
    --checkpoint "$CHECKPOINT" \
    --model-class "$MODEL_CLASS" \
    --module "$MODULE" \
    --input-length "$INPUT_LENGTH" \
    --num-classes "$NUM_CLASSES" \
    --output "$ONNX_OUT"

echo ""
echo "=== Step 2/3: Export to TFLite ==="
python src/export/export_tflite.py \
    --checkpoint "$CHECKPOINT" \
    --model-class "$MODEL_CLASS" \
    --module "$MODULE" \
    --input-length "$INPUT_LENGTH" \
    --num-classes "$NUM_CLASSES" \
    --output "$TFLITE_OUT"

echo ""
echo "=== Step 3/3: Validate exports ==="
python src/export/validate_export.py \
    --checkpoint "$CHECKPOINT" \
    --model-class "$MODEL_CLASS" \
    --module "$MODULE" \
    --input-length "$INPUT_LENGTH" \
    --num-classes "$NUM_CLASSES" \
    --onnx "$ONNX_OUT" \
    --tflite "$TFLITE_OUT"

echo ""
echo "=== Done ==="
echo "ONNX:   $ONNX_OUT ($(du -h "$ONNX_OUT" | cut -f1))"
echo "TFLite: $TFLITE_OUT ($(du -h "$TFLITE_OUT" | cut -f1))"
