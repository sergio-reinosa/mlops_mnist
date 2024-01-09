import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


if __name__ == "__main__":
    """Return train and test dataloaders for MNIST."""
    local_dir = "data/raw/corruptmnist"
    processed_dir = "data/processed/corruptedmnist"
    batch_size = 256

    train_files = [f for f in os.listdir(local_dir) if f.startswith("train_images")]
    test_files = [f for f in os.listdir(local_dir) if f.startswith("test_images")]

    # train
    train_images = torch.cat([(torch.load(f"{local_dir}/train_images_{i}.pt")) for i in range(len(train_files))])
    train_targets = torch.cat([torch.load(f"{local_dir}/train_target_{i}.pt") for i in range(len(train_files))])
    train_dataset = TensorDataset(train_images, train_targets)
    # trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test
    test_images = torch.load(f"{local_dir}/test_images.pt")
    test_targets = torch.load(f"{local_dir}/test_target.pt")
    test_dataset = TensorDataset(test_images, test_targets)
    # testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.save(train_dataset, os.path.join(processed_dir, "train_dataset.pt"))
    torch.save(test_dataset, os.path.join(processed_dir, "test_dataset.pt"))

    # return trainloader, testloader
