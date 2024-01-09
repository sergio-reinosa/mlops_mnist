import pytest
import os
import os.path
import torch
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def mnist_dataset():
    test_dataset = torch.load(os.path.join(_PATH_DATA, "test_dataset.pt"))
    train_dataset = torch.load(os.path.join(_PATH_DATA, "train_dataset.pt"))
    return train_dataset, test_dataset

def test_dataset_length():
    train_dataset, test_dataset = mnist_dataset()
    assert len(train_dataset) == 30000, "Training Dataset did not have the correct number of samples"
    assert len(test_dataset) == 5000, "Test Dataset did not have the correct number of samples"

def test_datapoint_shape():
    train_dataset, test_dataset = mnist_dataset()
    for dataset in [train_dataset, test_dataset]:
        for image, label in dataset:
            assert image.shape == (28, 28) or image.shape == (784,), "Imag shape was not correct"
            assert label >= 0 and label <= 9, "Class label was out of range"  
