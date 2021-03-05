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

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x.view(-1))])

dataset_train = datasets.MNIST(
    '~/mnist', 
    train=True, 
    download=True, 
    transform=transform)
dataset_valid = datasets.MNIST(
    '~/mnist', 
    train=False, 
    download=True, 
    transform=transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)

def train_info_gan(m, epoch = 20):
  for i in range(epoch):
    losses = []
    dis_loss = []
    gen_loss = []
    q_loss = []
    counter = 0
    for x, t in dataloader_train:
      counter += len(x)
      d, g, q = m.train(x)
      losses.append(sum([d, g, q]))
      dis_loss.append(d)
      gen_loss.append(g)
      q_loss.append(q)
      print(f'epoch{i}: {counter}/60000')
    plot_infogan(m, f'epoch_{i+1}')
    print(f'AVG loss {np.average(losses)}, dl {np.average(dis_loss)}, gl {np.average(gen_loss)}, ql {np.average(q_loss)}')
  return m

def plot_infogan(m, name):
  fig, axs = plt.subplots(1, 10, figsize=(30, 3))
  for i in range(0, 10): 
    fake = m.fake_x([i]).view(28, 28).detach().numpy()
    axs[i].imshow(fake, 'gray')
  plt.tight_layout()
  plt.savefig(f'{name}.png')
  plt.clf()
  plt.close('all')
  

def train(m, epoch = 20):
  for i in range(epoch):
    losses = []
    dis_loss = []
    gen_loss = []
    counter = 0
    for x, t in dataloader_train:
      counter += len(x)
      dl, gl = m.train(x, t)
      losses.append(dl + gl)
      dis_loss.append(dl)
      gen_loss.append(gl)
      print(f'epoch{i}: {counter}/60000')
    plot_generated(m, f'epoch_{i+1}')
    print(f'AVG loss {np.average(losses)}, dis_loss {np.average(dis_loss)}, gen_loss {np.average(gen_loss)}')
  return m

def plot_generated(m, name = 'dd'):
  fig, axs = plt.subplots(1, 10, figsize=(30, 3))
  for i in range(0, 10): 
    fake = m.generate([i]).view(28, 28).detach().numpy()
    axs[i].imshow(fake, 'gray')
  plt.tight_layout()
  plt.savefig(f'{name}.png')
  plt.clf()
  plt.close('all')

def test(model):
  fig = plt.figure(figsize=(10, 3))
  zs = []
  for x, t in dataloader_valid:
      # original
      for i, im in enumerate(x.view(-1, 28, 28).detach().numpy()[:10]):
        ax = fig.add_subplot(3, 10, i+1, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
      # generate from x
      y, z = model(x)
      zs.append(z)
      y = y.view(-1, 28, 28)
      for i, im in enumerate(y.cpu().detach().numpy()[:10]):
        ax = fig.add_subplot(3, 10, i+11, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
      # generate from z
      dds = [z[1] * (i * 0.1) + z[0] * ((9 - i) * 0.1) for i in range(10)]
      y2 = [model._decoder(dd).detach().view(28,28).numpy() for dd in dds]
      for i, im in enumerate(y2):
        ax = fig.add_subplot(3, 10, i+21, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
      break
  plt.savefig('dd.png')
  return fig

# ===========================

class GAN(nn.Module):
  def init_hook(self):
    pass

  def __init__(self):
    super().__init__()
    # self.fw1 = nn.Linear(28*28, 28*14)
    self.CEL = nn.CrossEntropyLoss()
    self.embedding = nn.Embedding(12, 240) # D, 0-9 used, 11 for UNK
    self.generator_layer1 = nn.Linear(480, 512) # G
    self.generator_layer2 = nn.Linear(512, 512) # G
    self.generator_layer3 = nn.Linear(512, 784) # G
    self.discriminator_common_layer1 = nn.Linear(1024, 512) # D
    self.discriminator_common_layer2 = nn.Linear(512, 256) # D
    self.discriminator_police = nn.Linear(256, 2) # D
    self.relu = nn.LeakyReLU(0.2)
    self.init_optim()
    self.init_hook()

  def generate(self, labels):
    labels = t.LongTensor(labels)
    return self.fake_x(self.embedding(labels))

  def init_optim(self):
    dis_params = chain(self.discriminator_common_layer1.parameters(), self.discriminator_common_layer2.parameters(), self.discriminator_police.parameters())
    self.optim_discriminator = optim.Adam(dis_params, lr=0.001)
    gen_params = chain(self.generator_layer1.parameters(), self.generator_layer2.parameters(), self.generator_layer3.parameters())
    self.optim_generator = optim.Adam(gen_params, lr=0.001)

  # label: (batch_size, 240)
  # return: # (batch_size, 784)
  def fake_x(self, label_emb):
    noises = t.randn(label_emb.shape[0], 480 - label_emb.shape[1])
    o = t.stack([t.cat((noise, emb)) for noise,emb in zip(noises,label_emb)]) # (batch_size, 480)
    o = self.relu(self.generator_layer1(o)) # (batch_size, 512) + active
    o = self.relu(self.generator_layer2(o)) # (batch_size, 512) + active
    o = self.generator_layer3(o) # (batch_size, 784)
    return o 

  # return loss
  # blend_x: (batch_size, 1024) float
  # marks: (batch_size) long
  def discriminate(self, blended_x, marks):
    o = self.relu(self.discriminator_common_layer1(blended_x)) # (batch_size, 512)
    o = self.relu(self.discriminator_common_layer2(o)) # (batch_size, 256)
    o = self.discriminator_police(o) # (batch_size, 2)
    return self.CEL(o, marks)

  def train_discriminator(self, xs, labels):
    labels = t.LongTensor(labels) # (batch_size)
    label_embs = self.embedding(labels) # (batch_size, 240)
    true_xs = t.stack([t.cat((x, label_emb)) for x,label_emb in zip(xs, label_embs)]) # (batch_size, 784 + 240)
    true_marks = t.ones(len(true_xs), dtype=t.long) # (batch_size)
    fake_xs = self.fake_x(label_embs) # (batch_size, 784)
    fake_xs = t.stack([t.cat((fake_x, label_emb)) for fake_x,label_emb in zip(fake_xs, label_embs)]) # (batch_size, 784 + 240)
    fake_marks = t.zeros(len(fake_xs), dtype=t.long)
    # Freezing generator and only train Discriminator for now
    dis_loss = self.discriminate(true_xs, true_marks) + self.discriminate(fake_xs, fake_marks)
    self.zero_grad()
    dis_loss.backward()
    self.optim_discriminator.step()
    return dis_loss

  def train_generator(self, labels):
    label_embs = self.embedding(labels) # (batch_size, 240)
    fake_xs = self.fake_x(label_embs) # (batch_size, 784)
    fake_xs = t.stack([t.cat((fake_x, label_emb)) for fake_x,label_emb in zip(fake_xs, label_embs)]) # (batch_size, 784 + 240)
    real_marks = t.ones(len(fake_xs), dtype=t.long)
    # Freezing Discriminator and only train Generator for now
    gen_loss = self.discriminate(fake_xs, real_marks)
    self.zero_grad()
    gen_loss.backward()
    self.optim_generator.step()
    return gen_loss

  # x: (batch_size, 784)
  # label: (batch_size)
  def train(self, xs, labels):
    dis_loss = self.train_discriminator(xs, labels)
    gen_loss = self.train_generator(labels)
    
    return dis_loss.detach().item(), gen_loss.detach().item()


