import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super(MyLightningModule, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16,8,3),
            nn.LeakyReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*20*20, 128),
            nn.Dropout(),
            nn.Linear(128,10)
        )
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        return self.classifier(self.model(x))

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

