import sys

import numpy as np
import pytest
import torch

from mlops_mnist.models.model import MyNeuralNetOne


def test_model():
    model = MyNeuralNetOne()
    image = np.ones((1, 28, 28))
    image = torch.from_numpy(image)
    images = image.view(image.shape[0], -1)
    log_ps = model(images.float())
    assert log_ps.shape == (1, 10)