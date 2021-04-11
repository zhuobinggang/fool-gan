import torch
t = torch
from itertools import chain
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import random as R
from importlib import reload
import ds


class InfoGAN(nn.Module):
  def __init__(self):
    super().__init__()
    self.noise_size = noise_size = self.category_embedding_size = 32
    self.img_size = img_size = 784
    self.BCE = nn.BCELoss()
    self.G = nn.Sequential(
      nn.ConvTranspose2d(noise_size + self.category_embedding_size, 128, 4, 1, 0, bias=False), # (batch, 128, 4, 4)
      nn.BatchNorm2d(128),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(128, 64, 4, 2, 0, bias=False), # (batch, 64, 10, 10)
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(64, 1, 3, 3, 1, bias=False), # (batch, 1, 28, 28)
      nn.Tanh(),
    )
    self.D = nn.Sequential(
      nn.Conv2d(1, 32, 7, 1, 0, bias=False), # (batch, 32, 22, 22)
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),
      nn.Conv2d(32, 64, 4, 2, 0, bias=False), # (batch, 64, 10, 10)
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
      nn.Conv2d(64, 128, 4, 2, 0, bias=False), # (batch, 128, 4, 4)
      nn.BatchNorm2d(128),
      nn.LeakyReLU(),
      nn.Conv2d(128, 1, 4, 1, 0, bias=False), # (batch, 1, 1, 1)
      nn.Sigmoid(),
    )
    self.init_optim()
    self.init_hook()

# ========================== Other Methods ==========================
def conv_out_size(W, K, S, P):
  return (((W - K + 2*P)/S) + 1)

def trans_conv_out_size(W, K, S, P):
  return (W - 1) * S - (2 * P) + K

