import matplotlib.pyplot as plt
import ds
import numpy as np

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

def plot_zero2nine(m, name = 'dd'):
  fig, axs = plt.subplots(1, 10, figsize=(30, 3))
  for i in range(0, 10): 
    fake = m.dry_run(i).view(28, 28).detach().cpu().numpy()
    axs[i].imshow(fake, 'gray')
  plt.tight_layout()
  plt.savefig(f'{name}.png')
  plt.clf()
  plt.close('all')

def train_and_output_infogan(m, epoch = 20, start = 1):
  for i in range(epoch):
    losses = []
    gls = []
    dls = []
    qls = []
    counter = 0
    for x, t in ds.dataloader_train:
      counter += len(x)
      gl, dl, ql = m.train(x)
      losses.append(gl + dl + ql)
      dls.append(dl)
      gls.append(gl)
      qls.append(ql)
      print(f'epoch{i}: {counter}/60000')
    plot_zero2nine(m, f'epoch_{i+start}')
    print(f'AVG loss {np.average(losses)}, dis_loss {np.average(dls)}, gen_loss {np.average(gls)}, q_loss {np.average(qls)}')
  return m
