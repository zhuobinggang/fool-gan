from gan import *
from importlib import reload

class GAN_Vanilla(nn.Module):
  def __init__(self):
    super().__init__()
    self.noise_size = noise_size = 64
    self.img_size = img_size = 784
    self.CEL = nn.CrossEntropyLoss()
    self.G = nn.Sequential(
      nn.Linear(noise_size, 256),
      nn.LeakyReLU(0.1),
      nn.Linear(256, int(img_size / 2)),
      nn.LeakyReLU(0.1),
      nn.Linear(int(img_size / 2), 784),
    )
    self.D = nn.Sequential(
      nn.Linear(img_size, int(img_size / 2)),
      nn.LeakyReLU(0.1),
      nn.Linear(int(img_size / 2), 2),
    )
    self.init_optim()
    self.init_hook()

  def init_hook(self):
    pass

  def init_optim(self):
    self.OptG = optim.AdamW(chain(self.G.parameters()), lr=0.001)
    self.OptD = optim.AdamW(chain(self.D.parameters()), lr=0.001)

  # return: # (batch, img_size)
  def fake_x(self, batch):
    noises = t.randn(batch, self.noise_size)
    return self.G(noises)

  # return loss
  # blend_x: (batch, img_size) float
  # marks: (batch) long
  def d(self, blended_x, marks):
    o = self.D(blended_x) # (batch_size, 2)
    return self.CEL(o, marks)

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

  # x: (batch_size, 784)
  # label: (batch_size)
  def train(self, xs):
    dl = self.td(xs)
    gl = self.tg(len(xs) * 2)
    return dl.detach().item(), gl.detach().item()


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
    for x, t in dataloader_train:
      counter += len(x)
      dl, gl = m.train(x)
      losses.append(dl + gl)
      dis_loss.append(dl)
      gen_loss.append(gl)
      print(f'epoch{i}: {counter}/60000')
    plot_random(m, f'epoch_{i+1}')
    print(f'AVG loss {np.average(losses)}, dis_loss {np.average(dis_loss)}, gen_loss {np.average(gen_loss)}')
  return m

