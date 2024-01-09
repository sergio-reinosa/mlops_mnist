import os

import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
from mlops_mnist.models.model import MyNeuralNetOne


def get_train_dataset(batch_size):
    processed_dir = "data/processed/corruptedmnist"
    batch_size = batch_size

    train_dataset = torch.load(os.path.join(processed_dir, "train_dataset.pt"))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


# THIS HYDRA THING IS NOT WORKING BC IT IS ALWAYS MESSING UP MY PATHS.......!!!!!!!!!!
# @hydra.main(config_path="../config", config_name="training_conf.yaml")
@click.command()
@click.option("--lr", default=0.001, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--model_name", default="trained_model.pt", help="name of the trained model")
def train(lr, epochs, batch_size, model_name):
    # project: name of project
    # entity: username or teamname
    wandb.init()
    """Train a model on MNIST."""
    model = MyNeuralNetOne()
    wandb.watch(model, log_freq=100)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set = get_train_dataset(batch_size)

    epochs = epochs
    # wandb.log({"image": wandb.Image('/home/hhauter/Documents/W23/MLOps/mlops_mnist/src/cat.jpeg')})
    for e in range(epochs):
        print(f"Epoch: {e}")
        running_loss = 0
        for images, labels in train_set:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            wandb.log({"loss": loss})
            print(f"Running Loss: {loss}")

    model_path = "models"
    torch.save(model, os.path.join(model_path, model_name))


if __name__ == "__main__":
    train()
