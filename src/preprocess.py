import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from config import DATA_DIR, REFERENCE_FILE, PROCESSED_DIR, TARGET_SIGNAL_LENGTH, RANDOM_SEED


# Map labels to integers
LABEL_MAP = {
    "N": 0,
    "A": 1,
    "O": 2,
    "~": 3
}


def load_ecg_signal(record_name):
    mat_path = os.path.join(DATA_DIR, record_name + ".mat")
    data = loadmat(mat_path)

    signal = data["val"][0].astype(np.float32)

    return signal


def normalize_signal(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal - mean

    return (signal - mean) / std


def pad_or_truncate(signal, target_length):
    current_length = len(signal)

    if current_length > target_length:
        start = (current_length - target_length) // 2
        end = start + target_length
        return signal[start:end]

    if current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(signal, (0, pad_width), mode="constant")

    return signal


def preprocess_signal(signal):

    signal = normalize_signal(signal)
    signal = pad_or_truncate(signal, TARGET_SIGNAL_LENGTH)

    return signal


def main():

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(REFERENCE_FILE, header=None)
    df.columns = ["record", "label"]

    signals = []
    labels = []

    print("Preprocessing ECG records...")

    for _, row in df.iterrows():

        record_name = row["record"]
        label = row["label"]

        signal = load_ecg_signal(record_name)
        signal = preprocess_signal(signal)

        signals.append(signal)
        labels.append(LABEL_MAP[label])

    X = np.array(signals, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print("Processed dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    # Train / temporary split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # Validation / test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )

    # Save processed datasets
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)

    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)

    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f, indent=4)

    print("\nSaved processed datasets to data/processed/")
    print("Train set:", X_train.shape, y_train.shape)
    print("Validation set:", X_val.shape, y_val.shape)
    print("Test set:", X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()