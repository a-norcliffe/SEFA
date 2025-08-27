"""Testing the utils for the Opportunistic Learning model."""

import unittest

import numpy as np
from math import ceil

import torch

from models.rl_utils import IterableIds, ExperienceBuffer



class TestIterator(unittest.TestCase):
  def test(self):
    data_size = 200
    batchsize = 7
    iterator = IterableIds(data_size, batchsize)

    # Create another set of indices, check they output all indices.
    ids1 = []
    for i in range(ceil(data_size/batchsize)):
      next_ids = iterator.next()
      if i != ceil(data_size/batchsize)-1:
        self.assertEqual(len(next_ids), batchsize)  # Check we get batchsize many ids.
      else:
        self.assertEqual(len(next_ids), data_size % batchsize)
      ids1.append(next_ids)
    ids1 = np.concatenate(ids1, axis=0)
    ids1_sorted = np.sort(ids1)
    self.assertTrue(np.all(ids1_sorted==np.arange(data_size)))  # Check we got the correct ids.
    
    # Create another set of indices, check they obey rules but are different.
    ids2 = []
    for i in range(ceil(data_size/batchsize)):
      next_ids = iterator.next()
      if i != ceil(data_size/batchsize)-1:
        self.assertEqual(len(next_ids), batchsize)  # Check we get batchsize many ids.
      else:
        self.assertEqual(len(next_ids), data_size % batchsize)
      ids2.append(next_ids)
    ids2 = np.concatenate(ids2, axis=0)
    ids2_sorted = np.sort(ids2)
    self.assertTrue(np.all(ids2_sorted==np.arange(data_size)))  # Check we got the correct ids.

    self.assertFalse(np.all(ids1==ids2))


class TestExperienceBuffer(unittest.TestCase):
  def test_adding_experience(self):
    buffer_size = 5
    experience_buffer = ExperienceBuffer(buffer_size)
    X = torch.arange(50).reshape(10, 5)
    y = torch.arange(10)
    m0 = torch.arange(50).reshape(10, 5)/10
    m1 = torch.arange(50).reshape(10, 5)/100
    r = -torch.arange(10)
    self.assertTrue(experience_buffer.length == 0)

    # Add experiences to the buffer.
    experience_buffer.append(X[0:1], y[0:1], m0[0:1], m1[0:1], r[0:1])
    self.assertTrue(experience_buffer.length == 1)

    experience_buffer.append(X[1:3], y[1:3], m0[1:3], m1[1:3], r[1:3])
    self.assertTrue(experience_buffer.length == 3)

    # Check the buffer is correct.
    self.assertTrue(torch.all(experience_buffer.x_buffer == X[0:3]))
    self.assertTrue(torch.all(experience_buffer.y_buffer == y[0:3]))
    self.assertTrue(torch.all(experience_buffer.m0_buffer == m0[0:3]))
    self.assertTrue(torch.all(experience_buffer.m1_buffer == m1[0:3]))
    self.assertTrue(torch.all(experience_buffer.r_buffer == r[0:3]))

    # Add even more experiences to the buffer.
    experience_buffer.append(X[3:], y[3:], m0[3:], m1[3:], r[3:])
    self.assertTrue(experience_buffer.length == buffer_size)  # Test it forgets things.

    # Check the buffer is correct.
    self.assertTrue(torch.all(experience_buffer.x_buffer == X[-buffer_size:]))
    self.assertTrue(torch.all(experience_buffer.y_buffer == y[-buffer_size:]))
    self.assertTrue(torch.all(experience_buffer.m0_buffer == m0[-buffer_size:]))
    self.assertTrue(torch.all(experience_buffer.m1_buffer == m1[-buffer_size:]))
    self.assertTrue(torch.all(experience_buffer.r_buffer == r[-buffer_size:]))

    # Check clearing the buffer, and adding too much initial experience.
    experience_buffer.clear_buffer()
    self.assertTrue(experience_buffer.length == 0)
    experience_buffer.append(X, y, m0, m1, r)
    self.assertTrue(experience_buffer.length == buffer_size)
    self.assertTrue(torch.all(experience_buffer.x_buffer == X[-buffer_size:]))
    self.assertTrue(torch.all(experience_buffer.y_buffer == y[-buffer_size:]))
    self.assertTrue(torch.all(experience_buffer.m0_buffer == m0[-buffer_size:]))
    self.assertTrue(torch.all(experience_buffer.m1_buffer == m1[-buffer_size:]))
    self.assertTrue(torch.all(experience_buffer.r_buffer == r[-buffer_size:]))

  def test_sampling_experience(self):
    # Can it sample randomly.
    buffer_size = 500
    experience_buffer = ExperienceBuffer(buffer_size)
    X = torch.arange(buffer_size)
    y = torch.arange(buffer_size)
    m0 = torch.arange(buffer_size)
    m1 = torch.arange(buffer_size)
    r = torch.arange(buffer_size)
    experience_buffer.append(X, y, m0, m1, r)

    # Sample from the buffer and check size.
    sample_size = 10
    X_sample, y_sample, m0_sample, m1_sample, r_sample = experience_buffer.sample(sample_size)
    self.assertTrue(X_sample.shape[0] == sample_size)
    self.assertTrue(y_sample.shape[0] == sample_size)
    self.assertTrue(m0_sample.shape[0] == sample_size)
    self.assertTrue(m1_sample.shape[0] == sample_size)
    self.assertTrue(r_sample.shape[0] == sample_size)

    # Check they are all the same.
    self.assertTrue(torch.all(X_sample == y_sample))
    self.assertTrue(torch.all(X_sample == m0_sample))
    self.assertTrue(torch.all(X_sample == m1_sample))
    self.assertTrue(torch.all(X_sample == r_sample))

    # Check they are not equal to something predictable.
    self.assertFalse(torch.all(X_sample == torch.arange(sample_size)))
    self.assertFalse(torch.all(X_sample == torch.arange(buffer_size-sample_size, buffer_size)))


if __name__ == "__main__":
  unittest.main()