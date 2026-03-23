import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED, PROCESSED_DIR, BASE_DIR
from dataset import ECGDataset
from model import ECGCNN


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def calculate_accuracy(outputs, labels):

    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()

    return correct / labels.size(0)


def evaluate(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0
    running_acc = 0

    with torch.no_grad():

        for signals, labels in dataloader:

            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)

            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            running_loss += loss.item()
            running_acc += acc

    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)

    return avg_loss, avg_acc


def main():

    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = ECGDataset("train")
    val_dataset = ECGDataset("val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(os.path.join(BASE_DIR, "outputs", "models"), exist_ok=True)

    best_model_path = os.path.join(BASE_DIR, "outputs", "models", "best_model.pth")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0
        running_acc = 0

        for signals, labels in train_loader:

            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(signals)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            acc = calculate_accuracy(outputs, labels)

            running_loss += loss.item()
            running_acc += acc

        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

            print("Best model saved.")

    print("\nTraining complete.")
    print("Best model:", best_model_path)


if __name__ == "__main__":
    main()