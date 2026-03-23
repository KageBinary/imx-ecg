import torch
import numpy as np
import os
from model import ECGCNN
from config import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = ECGCNN()

# Load saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

print("Model loaded successfully")