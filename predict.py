import os
import numpy as np
import torch
import pytorch_lightning as pl
from dataset import ISICLoader
from metrics import iou_score, dice_score

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset

from models.AC_MambaSeg import AC_MambaSeg

model = AC_MambaSeg()
DATA_PATH = ''
CHECKPOINT_PATH = ''

# Lightning module
class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        metrics = {"Test Dice": dice, "Test Iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics
    
model.eval()

# Dataset & Data Loader
data = np.load(DATA_PATH)
x_test, y_test = data["image"], data["mask"]
test_dataset = DataLoader(ISICLoader(x_test, y_test, typeData="test"), batch_size=1, num_workers=2, prefetch_factor=16)

# Prediction
trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)
trainer.test(segmentor, test_dataset)
