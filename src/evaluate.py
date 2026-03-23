import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset import ECGDataset
from model import ECGCNN
from config import BASE_DIR, BATCH_SIZE


CLASS_NAMES = ["N", "A", "O", "~"]


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_path = os.path.join(BASE_DIR, "outputs", "models", "best_model.pth")

    if not os.path.exists(model_path):
        print("Error: best_model.pth not found")
        return

    # Create model
    model = ECGCNN()

    # Load saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("\nModel loaded successfully\n")

    print("Model architecture:\n")
    print(model)

    # Load test dataset
    test_dataset = ECGDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for signals, labels in test_loader:

            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)

    print("\nTest Accuracy:", acc)

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()