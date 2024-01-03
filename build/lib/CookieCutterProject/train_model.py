import torch
from torch import nn
from torch import optim
from models.model import MyNeuralNet
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

def get_train_dataset():
    processed_dir = "/home/hhauter/Documents/W23/MLOps/CookieCutterProject/data/processed/corruptedmnist"
    batch_size = 256

    # Load and transform train data
    train_dataset = torch.load(os.path.join(processed_dir, "train_dataset.pt"))
    test_dataset = torch.load(os.path.join(processed_dir, "test_dataset.pt"))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def train():
    """Train a model on MNIST."""
    model = MyNeuralNet()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_set, _ = get_train_dataset()

    epochs = 20

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    model_path = "/home/hhauter/Documents/W23/MLOps/CookieCutterProject/models"
    torch.save(model, os.path.join(model_path, 'trained_model.pt'))

if __name__ == '__main__':
    train()