import torch
import os
from torch.utils.data import DataLoader, TensorDataset

from data.get_dataset import get_test_dataset


def visualize():
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")

    # TODO: Implement evaluation logic here
    model = torch.load("/home/hhauter/Documents/W23/MLOps/CookieCutterProject/models/trained_model.pt")
    test_set = get_test_dataset()

    equals = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_set:
            outputs = torch.exp(model(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            equals += (predicted == labels).sum().item()
    print(f"Accuracy: {(equals/total)*100}%")
