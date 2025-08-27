"""Utils for the RL models. Includes iterable ids so we don't need a data
loader and an experience buffer."""

from math import ceil, floor

import numpy as np

import torch


class IterableIds:
  """Iterable indices so that we can iterate through the dataset, without
  using a data loader.
  """
  def __init__(self, data_len, batchsize):
    self.data_len = data_len
    self.batchsize = batchsize
    self.shuffle_ids()

  def shuffle_ids(self):
    self.shuffled = np.random.permutation(self.data_len)
    self.i = 0

  def next(self):
    if self.i == ceil(self.data_len/self.batchsize):
      self.shuffle_ids()
      ids = self.shuffled[self.i*self.batchsize:(self.i+1)*self.batchsize]
      self.i += 1
      return ids
    elif self.i == floor(self.data_len/self.batchsize):
      ids = self.shuffled[self.i*self.batchsize:]
      self.shuffle_ids()
      return ids
    else:
      ids = self.shuffled[self.i*self.batchsize:(self.i+1)*self.batchsize]
      self.i += 1
      return ids


class ExperienceBuffer:
  """Experience buffer for RL training. We keep track of (s, r, a, s1, y)
  state, reward, action, next state, and target.
  We do this by tracking a list for each of these:
  1. x
  2. y
  3. mask
  4. next mask
  5. reward
  Since the two different masks tells us what action was taken.
  """
  def __init__(self, buffer_size):
    self.x_buffer = None
    self.y_buffer = None
    self.m0_buffer = None
    self.m1_buffer = None
    self.r_buffer = None
    self.buffer_size = buffer_size

  def append(self, x, y, m0, m1, r):
    if self.x_buffer is None:
      self.x_buffer = x[-self.buffer_size:]
      self.y_buffer = y[-self.buffer_size:]
      self.m0_buffer = m0[-self.buffer_size:]
      self.m1_buffer = m1[-self.buffer_size:]
      self.r_buffer = r[-self.buffer_size:]
    else:
      self.x_buffer = torch.cat([self.x_buffer, x], dim=0)[-self.buffer_size:]
      self.y_buffer = torch.cat([self.y_buffer, y], dim=0)[-self.buffer_size:]
      self.m0_buffer = torch.cat([self.m0_buffer, m0], dim=0)[-self.buffer_size:]
      self.m1_buffer = torch.cat([self.m1_buffer, m1], dim=0)[-self.buffer_size:]
      self.r_buffer = torch.cat([self.r_buffer, r], dim=0)[-self.buffer_size:]

  @property
  def length(self):
    if self.x_buffer is None:
      return 0
    return self.x_buffer.shape[0]

  def sample(self, batchsize):
    # NOTE: Length of buffer must be larger than the batchsize we sample.
    assert self.length >= batchsize, "Cannot sample more than buffer length."
    ids = np.random.choice(a=self.length, size=batchsize, replace=False)
    x = self.x_buffer[ids]
    y = self.y_buffer[ids]
    m0 = self.m0_buffer[ids]
    m1 = self.m1_buffer[ids]
    r = self.r_buffer[ids]
    return x, y, m0, m1, r

  def clear_buffer(self):
    self.x_buffer = None
    self.y_buffer = None
    self.m0_buffer = None
    self.m1_buffer = None
    self.r_buffer = None

  def loop_through_buffer(self, batchsize):
    assert self.x_buffer is not None
    shuffled_ids = np.random.permutation(self.length)
    shuffled_ids = np.array_split(shuffled_ids, ceil(self.length/batchsize))
    for ids in shuffled_ids:
      x = self.x_buffer[ids]
      y = self.y_buffer[ids]
      m0 = self.m0_buffer[ids]
      m1 = self.m1_buffer[ids]
      r = self.r_buffer[ids]
      yield x, y, m0, m1, r