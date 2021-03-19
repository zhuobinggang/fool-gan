from gan2 import *

class DCGAN(GAN_Vanilla):
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

