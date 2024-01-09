import torch
import os
from torch.utils.data import DataLoader, TensorDataset


def get_test_dataset():
    processed_dir = "data/processed/corruptedmnist"
    batch_size = 256

    test_dataset = torch.load(os.path.join(processed_dir, "test_dataset.pt"))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_dataloader
