import torch as t
nn = t.nn
import random as R
import numpy as np
from torch import optim
from importlib import reload
import gan as G
from itertools import chain

class InfoGAN(nn.Module):
  def __init__(self):
    super().__init__()
    # self.fw1 = nn.Linear(28*28, 28*14)
    self.CEL = nn.CrossEntropyLoss()
    self.input_size = 64
    self.c_ember = nn.Embedding(12, self.input_size) # D, 0-9 used, 11 for UNK
    self.G = nn.Sequential(
      nn.Linear(self.input_size * 2, 256), # G
      nn.LeakyReLU(0.1),
      nn.Linear(256, 384), # G
      nn.LeakyReLU(0.1),
      nn.Linear(384, 784), # G
    )
    self.D_common = nn.Sequential(
      nn.Linear(784, 256),
      nn.LeakyReLU(0.1),
    )
    self.D = nn.Sequential(
      nn.Linear(256, 128),
      nn.LeakyReLU(0.1),
      nn.Linear(128, 2),
    )
    self.Q = nn.Sequential(
      nn.Linear(256, 128),
      nn.LeakyReLU(0.1),
      nn.Linear(128, 10),
    )
    self.init_optim()

  def init_optim(self):
    lr = 1e-3
    self.optim_D = optim.Adam(chain(self.D_common.parameters(), self.D.parameters()), lr=lr)
    self.optim_G = optim.Adam(chain(self.G.parameters(), self.Q.parameters()), lr=lr)

  # return: # (batch_size, 784)
  def fake_x(self, c_ids):
    batch = len(c_ids)
    c = self.c_ember(t.LongTensor(c_ids))
    z = t.randn(batch, self.input_size) # (batch, input_size)
    latent_code = t.stack([t.cat((z,c)) for z,c in zip(z, c)]) # (batch, input_size * 2)
    return self.G(latent_code) # (batch, 784)

  def rand_c_labels(self, batch):
    ids = []
    for i in range(0, batch):
      ids.append(R.randint(0,9))
    return ids

  # return loss
  # xs: (batch, 784) float
  # marks: (batch_size) long
  def D_loss(self, blend_xs, labels):
    return self.CEL(self.D(self.D_common(blend_xs)), labels)

  def train_D(self, xs):
    batch = len(xs)
    true_xs_loss = self.D_loss(xs, t.ones(batch, dtype=t.long))
    fake_xs_loss = self.D_loss(self.fake_x(self.rand_c_labels(batch)), t.zeros(batch, dtype=t.long))
    loss = true_xs_loss + fake_xs_loss
    self.zero_grad()
    loss.backward()
    self.optim_D.step()
    return loss.detach().item()

  def train_G_Q(self, batch):
    c_labels = self.rand_c_labels(batch) # (batch) long
    fake_xs = self.fake_x(c_labels)
    g_loss = self.D_loss(fake_xs, t.ones(batch, dtype=t.long))
    q_loss = self.CEL(self.Q(self.D_common(fake_xs)), t.LongTensor(c_labels))
    loss = g_loss + q_loss
    self.zero_grad()
    loss.backward()
    self.optim_G.step()
    return g_loss.detach().item(), q_loss.detach().item()

  # xs : (batch ,784)
  def train(self, xs):
    d = self.train_D(xs)
    g,q = self.train_G_Q(len(xs))
    return d, g, q


