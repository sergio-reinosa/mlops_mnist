import os

import pytest 
from mlops_mnist.train_model import train


@pytest.mark.parametrize("lr,ep,bt", [(0.001, 1, 64), (0.01, 1, 32), (0.0001, 1, 128)])
def test_training(lr, ep, bt):
    train(lr,ep,bt)
    path = os.path.join("models", "trained_model.pt")
    assert os.path.exists(path)