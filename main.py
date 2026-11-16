# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub

from config import NUM_CLASSES, LEARNING_RATE, BEST_MODEL_PATH
from dataset import prepare_dataloaders
from models import CNNMambaHybrid
from train import resume_training_fresh
from evaluate import evaluate_model, print_metrics
from utils import get_device


def main():
    device = get_device()

    # Dataset download (Kaggle)
    path = kagglehub.dataset_download("obulisainaren/multi-cancer")
    root_dir = os.path.join(path, "Multi Cancer", "Multi Cancer")

    train_loader, val_loader, test_loader, full_dataset = prepare_dataloaders(root_dir)

    model = CNNMambaHybrid(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Optional: train / resume
    # train_losses, val_losses, train_accuracies, val_accuracies = resume_training_fresh(
    #     model, train_loader, val_loader, criterion, optimizer,
    #     additional_epochs=5, device=device
    # )

    # Load best model
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        model.eval()
        print("Loaded best model weights.")
    else:
        print("Best model not found. Train first.")

    # Evaluate
    test_accuracy, test_predictions, test_labels = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print_metrics(test_labels, test_predictions)


if __name__ == "__main__":
    main()