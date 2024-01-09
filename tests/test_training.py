import os
from click.testing import CliRunner
import pytest 
from mlops_mnist.train_model import train


@pytest.mark.parametrize("lr,epochs,batch_size,model_name", [(0.001, 1, 64, "test.pt"), (0.01, 1, 32, "unicorn.pt"), (0.0001, 1, 128, "helloworld.pt")])
def test_training(lr, epochs, batch_size, model_name):
    runner = CliRunner()
    runner.invoke(train, ["--lr", str(lr), "--epochs", str(epochs), "--batch_size", str(batch_size), "--model_name", model_name])

    path = os.path.join("/home/hhauter/Documents/W23/MLOps/mlops_mnist/models", model_name)
    assert os.path.exists(path)

    # clean up 
    os.remove(path) 