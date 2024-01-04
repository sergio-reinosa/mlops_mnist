import torch
from torch import nn
from torch import optim
from models.model import MyNeuralNet
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


def get_train_dataset():
    processed_dir = "data/processed/corruptedmnist"
    batch_size = 256

    train_dataset = torch.load(os.path.join(processed_dir, "train_dataset.pt"))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


def train():
    """Train a model on MNIST."""
    model = MyNeuralNet()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_set = get_train_dataset()

    epochs = 20

    for e in range(epochs):
        print(f'Epoch: {e}')
        running_loss = 0
        for images, labels in train_set:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f'Running Loss: {running_loss}')

    model_path = "models"
    torch.save(model, os.path.join(model_path, "trained_model.pt"))


if __name__ == "__main__":
    train()
