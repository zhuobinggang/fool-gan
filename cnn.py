import torch as t
import logging
nn = t.nn

class ImgSelfAtt(nn.Module):
  def __init__(self, size = 3):
    super().__init__()
    self.size = size
    self.selfatt = nn.TransformerEncoderLayer(size * size, 1, int(size * size * 1.5), 0)

  # x: (batch, H, W)
  def forward(self, x):
    if not ((x.shape[1] / self.size).is_integer() and (x.shape[2] / self.size).is_integer()):
      logging.warning('Wrong shape input: {x.shape}')
    else:
      # TODO:

