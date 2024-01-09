import torch
import os
from torch.utils.data import DataLoader, TensorDataset

from data.get_dataset import get_test_dataset


def evaluate():
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")

    # TODO: Implement evaluation logic here
    model_path = "models"
    model = torch.load(os.path.join(model_path,"trained_model.pt"))
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


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


if __name__ == "__main__":
    evaluate()
