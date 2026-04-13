import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from config import DATA_DIR, REFERENCE_FILE

def inspect_dataset():

    print("Loading labels...")
    df = pd.read_csv(REFERENCE_FILE, header=None)
    df.columns = ["record", "label"]

    print("\nTotal records:", len(df))

    lengths = []
    missing = []

    for record in df["record"]:

        mat_path = os.path.join(DATA_DIR, record + ".mat")

        if not os.path.exists(mat_path):
            missing.append(record)
            continue

        data = loadmat(mat_path)
        signal = data["val"][0]

        lengths.append(len(signal))

    print("\nSignal length statistics")

    print("Min length:", np.min(lengths))
    print("Max length:", np.max(lengths))
    print("Mean length:", int(np.mean(lengths)))

    print("\nClass distribution:")
    print(df["label"].value_counts())

    if missing:
        print("\nMissing files:", missing)

if __name__ == "__main__":
    inspect_dataset()