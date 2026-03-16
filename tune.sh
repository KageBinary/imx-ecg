#!/bin/bash

# Create a directory to store all the text output so you don't lose it
mkdir -p logs
mkdir -p checkpoints

# Define the values you want to test
WIDTHS=(8 16 32)
KERNELS=(3 5 7)
HIDDENS=(64 128)

echo "Starting Hyperparameter Sweep..."

# Loop through every combination
for w in "${WIDTHS[@]}"; do
  for k in "${KERNELS[@]}"; do
    for h in "${HIDDENS[@]}"; do
      
      # Create a unique name for this specific test
      RUN_NAME="w${w}_k${k}_h${h}"
      CHECKPOINT="checkpoints/model_${RUN_NAME}.pt"
      LOG_FILE="logs/log_${RUN_NAME}.txt"
      
      echo "-------------------------------------------------"
      echo "Training Model: Base Width=$w | Kernel=$k | Hidden=$h"
      echo "Saving to: $CHECKPOINT"
      
      # Run your Python training script with the variables
      # We use a smaller subset (e.g., 500) and fewer epochs (e.g., 5) for a faster sweep
      python src/train_fft_gp.py \
        --data-dir data2017 \
        --base-width $w \
        --kernel-size $k \
        --hidden-dim $h \
        --epochs 5 \
        --subset 500 \
        --cpu \
        --checkpoint-path $CHECKPOINT > $LOG_FILE 2>&1
        
      echo "Finished $RUN_NAME. Check $LOG_FILE for results."
      
    done
  done
done

echo "All sweeps completed!"