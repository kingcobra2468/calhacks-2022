#Model
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

#Dataloader
from torch.utils.data import DataLoader, Dataset

#Dataset
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
import os

import requests
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import boto3
import joblib

unhealthy = ['apple_pie', 
             'baby_back_ribs',
             'grilled_cheese_sandwich',
             'baklava',
             'carot_cake',
             'beef_carpaccio',
             'beignets',
             'bread_pudding',
             'breakfast_burrito',
             'cannoli',
             'carrot_cake',
             'cheesecake',
             'chicken_wings',
             'chicken_quesadilla',
             'chocolate_cake',
             'chocolate_mousse',
             'churros',
             'creme_brulee',
             'croque_madame',
             'cup_cakes',
             'donuts',
             'filet_mignon',
             'french_fries',
             'grilled_cheese_sandwich',
             'hamburger',
             'hot_dog',
             'ice_cream',
             'macaroni_and_cheese',
             'macarons',
             'nachos',
             'onion_rings',
             'pancakes',
             'panna_cotta',
             'peking_duck',
             'pizza',
             'poutine',
             'prime_rib',
             'pulled_work_sandwich',
             'red_velvet_cake',
             'steak',
             'strawberry_shortcake',
             'tacos',
             'tiramisu',
             'waffles',
             'beef_tartare',
             'pork_chop',
             'pulled_pork_sandwich'
             ]
def download_data():
    os.system('wget http://da?ta.vision.ee.ethz.ch/cvl/food-101.tar.gz')
    os.system('tar -xf /content/food-101.tar.gz')
def build_df():
    classes = os.listdir('/content/food-101/images/')
    dfFull = pd.DataFrame()
    imagedir = './food-101/images/'
    for cls in classes:
        images = map(lambda x: os.path.join(cls, x), os.listdir(os.path.join(imagedir, cls)))
        df = pd.DataFrame()
        df['images'] = pd.Series(images)
        df['class'] = [cls]* len(df['images'])
        if cls in unhealthy:
            df['label'] = [0]* len(df['images'])
        else:
            df['label'] = [1] * len(df['images'])
    dfFull = pd.concat([dfFull, df], axis = 0, ignore_index=True)
    return dfFull
IMAGEDIR = './food-101/images/'

class Transforms(nn.Sequential):
    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self.transforms = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        )
    @torch.no_grad()
    def forward(self, x):
        return self.transforms(x)
dfFull = pd.read_csv('healthy_dataset.csv')
class CNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.model = models.resnext101_64x4d(weights = models.ResNeXt101_64X4D_Weights.DEFAULT)
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Sequential(
               nn.Linear(num_ftrs, 128),
               nn.Sigmoid(),
               nn.Linear(128, 2))
  def forward(self, x):
    return self.model(x).squeeze(-1) 

class FoodDataset(Dataset):
  def __init__(self, df: pd.DataFrame, imgdir, transforms = None):
    self.df = df
    self.imgdir = imgdir
    self.transforms = transforms
  def __len__(self):
    return len(self.df)
  def __getitem__(self, idx):
    if torch.torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.df.iloc[idx]['images']
    img_name = os.path.join(self.imgdir, img_name)
    image = io.imread(img_name)
    pred = self.df.iloc[idx]['label']
    if self.transforms:
      image = self.transforms(image)
    return image, torch.as_tensor(pred, dtype = torch.float32)
class LightningCNN(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = models.resnext101_64x4d(weights = models.ResNeXt101_64X4D_Weights.DEFAULT)
    num_ftrs = self.model.fc.in_features
    for param in self.model.parameters():
      param.requires_grad = False
    self.model.fc = nn.Sequential(
              nn.Linear(num_ftrs, 128),
              nn.Sigmoid(),
              nn.Linear(128, 2))
  def forward(self, x):
    return self.model(x).squeeze(-1)
  def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y
  def configure_optimizers(self, lr=1e-4, epochs=50):  #Se tienen que poner los valores que correspondan a lo que queremos despues
        learning_rate = lr
        optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs/2)
        return [optimizer],[scheduler]
  def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        return {'loss': loss}
  def test_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        test_loss = 0
        test_loss += self.loss_fn(pred, y).item()
        test_loss /= len(batch) #num_batches
        #print(f"Avg test loss: {test_loss:>8f} \n")
        return {'test loss': test_loss}
  def prepare_data(self):
        #this happend only one time (download data)
        if not os.path.exists(IMAGEDIR):
            download_data()
  def setup(self, stage=None):
        #this runs in every single GPU
        self.trn = FoodDataset(dfFull, IMAGEDIR, self.preprocess)
        self.tst = FoodDataset(dfFull, IMAGEDIR, self.preprocess)

def train():
  model = LightningCNN()
  trainer = pl.Trainer(gpus=1)
  trainer.fit(model)