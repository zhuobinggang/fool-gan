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

class DCGAN(nn.Module):
  def __init__(self):
    super().__init__()
    self.noise_size = noise_size = 64
    self.img_size = img_size = 784
    self.BCE = nn.BCELoss()
    self.G = nn.Sequential(
      nn.ConvTranspose2d(noise_size, 128, 4, 1, 0, bias=False), # (batch, 128, 4, 4)
      nn.BatchNorm2d(128),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(128, 64, 4, 2, 0, bias=False), # (batch, 64, 10, 10)
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(64, 1, 3, 3, 1, bias=False), # (batch, 1, 28, 28)
      # nn.ConvTranspose2d(64, 32, 4, 2, 0, bias=False), # (batch, 32, 22, 22)
      # nn.BatchNorm2d(32),
      # nn.LeakyReLU(),
      # nn.ConvTranspose2d(32, 1, 7, 1, 0, bias=False), # (batch, 1, 28, 28)
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


  def init_hook(self):
    self.W = 28
    self.H = 28
    self.C = 1

  def init_optim(self):
    self.OptG = optim.AdamW(chain(self.G.parameters()), lr=0.001)
    self.OptD = optim.AdamW(chain(self.D.parameters()), lr=0.001)

  # return loss
  # blend_x: (batch, img_size) float
  # marks: (batch) long
  def d(self, blended_x, marks):
    o = self.D(blended_x) # (batch_size, 1)
    return self.BCE(o.view(-1), marks.float())

  def td(self, xs):
    loss = self.d(xs, t.ones(len(xs), dtype=t.long))
    loss += self.d(self.fake_x(len(xs)), t.zeros(len(xs), dtype=t.long))
    self.zero_grad()
    loss.backward()
    self.OptD.step()
    return loss

  def tg(self, batch):
    loss = self.d(self.fake_x(batch), t.ones(batch, dtype=t.long))
    self.zero_grad()
    loss.backward()
    self.OptG.step()
    return loss

  # return: # (batch, img_size)
  def fake_x(self, batch):
    noises = t.randn(batch, self.noise_size, 1, 1)
    dd = self.G(noises)
    return dd

  # x: (batch_size, 784)
  # label: (batch_size)
  def train(self, xs):
    xs = xs.view(-1, self.C, self.H, self.W)
    dl = self.td(xs)
    gl = self.tg(len(xs))
    return dl.detach().item(), gl.detach().item()


# ================================== scripts =====================================

def plot_random(m, name = 'dd'):
  fig, axs = plt.subplots(1, 10, figsize=(30, 3))
  for i in range(0, 10): 
    fake = m.fake_x(1).view(28, 28).detach().numpy()
    axs[i].imshow(fake, 'gray')
  plt.tight_layout()
  plt.savefig(f'{name}.png')
  plt.clf()
  plt.close('all')

def train_vanilla(m, epoch = 20):
  for i in range(epoch):
    losses = []
    dis_loss = []
    gen_loss = []
    counter = 0
    for x, t in ds.dataloader_train:
      counter += len(x)
      dl, gl = m.train(x)
      losses.append(dl + gl)
      dis_loss.append(dl)
      gen_loss.append(gl)
      print(f'epoch{i}: {counter}/60000')
    plot_random(m, f'epoch_{i+1}')
    print(f'AVG loss {np.average(losses)}, dis_loss {np.average(dis_loss)}, gen_loss {np.average(gen_loss)}')
  return m

