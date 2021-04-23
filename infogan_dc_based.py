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
import random
from importlib import reload
import ds


GPU_OK = torch.cuda.is_available()
device = "cuda" if GPU_OK else "cpu"

class InfoGAN(nn.Module):
  def __init__(self):
    super().__init__()
    self.noise_size = noise_size = self.category_embedding_size = 32
    self.img_size = img_size = 784
    self.BCE = nn.BCELoss()
    self.CEL = nn.CrossEntropyLoss()
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
    self.D_share = nn.Sequential(
      nn.Conv2d(1, 32, 7, 1, 0, bias=False), # (batch, 32, 22, 22)
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),
      nn.Conv2d(32, 64, 4, 2, 0, bias=False), # (batch, 64, 10, 10)
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
      nn.Conv2d(64, 128, 4, 2, 0, bias=False), # (batch, 128, 4, 4)
      nn.BatchNorm2d(128),
      nn.LeakyReLU()
    )
    self.D = nn.Sequential(
      self.D_share,
      nn.Conv2d(128, 2, 4, 1, 0, bias=False), # (batch, 2, 1, 1)
    )
    self.Q = nn.Sequential(
      nn.Conv2d(128, 24, 4, 1, 0, bias=False), # (batch, 24, 1, 1)
      nn.BatchNorm2d(24),
      nn.LeakyReLU(),
      nn.Flatten(), # (batch, 24)
      nn.Linear(24, 10), # (batch, 10)
    )
    self.c_ember = nn.Embedding(10, self.category_embedding_size)
    self.init_optim()
    self = self.cuda() if GPU_OK else self

  def init_optim(self):
    lr = 1e-3
    self.optim_D = optim.AdamW(self.D.parameters(), lr=lr)
    self.optim_G = optim.AdamW(self.G.parameters(), lr=lr)
    self.optim_Q = optim.AdamW(chain(self.Q.parameters(), self.c_ember.parameters()), lr=lr)

  def processed_xs(self, xs):
    batch, feature = xs.shape
    return xs.view(batch, 1, 28, 28)

  def random_category(self):
    return random.randint(0, 9)

  def cs_from_category(self, category, batch):
    temp = t.LongTensor([category]).repeat(batch).to(device) # (batch)
    return self.c_ember(temp) # (batch, category_embedding_size)

  def cs_cat_zs(self, cs): # (batch, noise_size + category_embedding_size, 1, 1)
    batch, _ = cs.shape
    noise = t.randn(batch, self.noise_size).to(device)
    o = t.cat((cs, noise), 1)
    return o.view(batch, self.noise_size + self.category_embedding_size, 1, 1)

  def train_D(self, xs, label): 
    batch = xs.shape[0]
    o = self.D(xs).view(batch, 2) # (batch, 2)
    labels = t.LongTensor([label]).repeat(batch).to(device) # (batch)
    return self.CEL(o, labels)

  def train_Q(self, xs, category):
    batch, _, _, _ = xs.shape
    o = self.D_share(xs) # (batch, 10)
    o = self.Q(o)
    labels = t.LongTensor([category]).repeat(batch).to(device)
    return self.CEL(o, labels)
 
  def train(self, xs):
    xs = xs.to(device)
    xs = self.processed_xs(xs) # (batch, 1, 28, 28)
    batch = xs.shape[0]
    random_category = self.random_category()
    cs = self.cs_from_category(random_category, batch) # (batch, category_embedding_size)
    cs_cat_zs = self.cs_cat_zs(cs) # (batch, noise_size + category_embedding_size, 1, 1)

    # Train D
    fake_xs = self.G(cs_cat_zs.detach()) # (batch, 1, 28, 28)
    assert fake_xs.shape == xs.shape
    d_loss = self.train_D(xs, 1) + self.train_D(fake_xs, 0)
    self.zero_grad()
    d_loss.backward()
    self.optim_D.step()

    # Train G 
    fake_xs = self.G(cs_cat_zs.detach()) # (batch, 1, 28, 28)
    g_loss = self.train_D(fake_xs, 1)
    self.zero_grad()
    g_loss.backward()
    self.optim_G.step()

    # Train Q
    fake_xs = self.G(cs_cat_zs.detach()) # (batch, 1, 28, 28)
    q_loss = self.train_Q(fake_xs, random_category)
    self.zero_grad()
    q_loss.backward()
    self.optim_Q.step()

    return g_loss.item(), d_loss.item(), q_loss.item()

  def dry_run(self, category): # (28 * 28)
    cs = self.cs_from_category(category, 1) # (batch, category_embedding_size)
    cs_cat_zs = self.cs_cat_zs(cs) # (batch, noise_size + category_embedding_size, 1, 1)
    fake_xs = self.G(cs_cat_zs.detach()) # (batch, 1, 28, 28)
    return fake_xs.view(28 * 28)

# ========================== Other Methods ==========================
def conv_out_size(W, K, S, P):
  return (((W - K + 2*P)/S) + 1)

def trans_conv_out_size(W, K, S, P):
  return (W - 1) * S - (2 * P) + K

