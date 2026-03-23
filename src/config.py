import os
#====================================================================
# 1) Run python src/train.py
# 2) Run python src/evaluate.py

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw dataset location
DATA_DIR = os.path.join(BASE_DIR, "data", "training")
REFERENCE_FILE = os.path.join(DATA_DIR, "REFERENCE.csv")

# Processed dataset location
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# ECG preprocessing settings
TARGET_SIGNAL_LENGTH = 10000

# Random seed
RANDOM_SEED = 42

# Number of output classes
NUM_CLASSES = 4