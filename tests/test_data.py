
import os
import os.path

import pytest
import torch

import os.path

from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA,"train_dataset.pt")), reason="Data files not found")
def test_data_training():
    train_dataset = torch.load(os.path.join(_PATH_DATA,"train_dataset.pt"))
    

    assert len(train_dataset) == 30000, "training dataset does not match the size"

    for image, label in train_dataset:
            assert image.shape == (28, 28) or image.shape == (784,), "Imag shape was not correct"
            assert label >= 0 and label <= 9, "Class label was out of range"

@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA,"test_dataset.pt")), reason="Data files not found")
def test_data_test():

    test_dataset = torch.load(os.path.join(_PATH_DATA,"test_dataset.pt"))

    assert len(test_dataset) == 5000, "test dataset does not match the size" 

    for image, label in test_dataset:
            assert image.shape == (28, 28) or image.shape == (784,), "Imag shape was not correct"
            assert label >= 0 and label <= 9, "Class label was out of range"

