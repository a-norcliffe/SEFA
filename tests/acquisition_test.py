"""Test to check that acquisition from the base model works as expected.
We set the scores to be a fixed order, where the last feature is best.
"""


import unittest

import torch
import torch.nn as nn

from models.base import BaseModel


class TestModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)
    # Used so we have a parameter to find the device.
    self.dummy_param = nn.Parameter(torch.zeros(1))

  def calculate_acquisition_scores(self, x, mask):
    # We set the scores to be a fixed order, where the last feature is best.
    # This is to test that the acquisition works as expected.
    fixed_order_scores = torch.arange(mask.shape[-1], device=self.device, dtype=torch.float32)
    return fixed_order_scores.repeat(mask.shape[0], 1)


class TestAcquisition(unittest.TestCase):

  def test_no_missing_data(self):
    config = {
      "num_con_features": 5,
      "num_cat_features": 0,
      "most_categories": 0,
      "out_dim": 10,
      "max_dim": None
    }
    model = TestModel(config)

    mask_acq = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0],
      [1.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    x = torch.randn_like(mask_acq)
    mask_data = torch.ones_like(mask_acq)

    # First acquisition.
    mask_acq_next = model.acquire(x, mask_acq, mask_data)
    expected_mask_acq_next = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 1.0, 1.0],
      [1.0, 0.0, 1.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    self.assertTrue(torch.all(mask_acq_next == expected_mask_acq_next))
    change = torch.sum(mask_acq_next - mask_acq, dim=-1)
    self.assertTrue(torch.all(change == torch.tensor([1.0, 1.0, 1.0, 1.0])))
    mask_acq = mask_acq_next

    # Second acquisition.
    mask_acq_next = model.acquire(x, mask_acq, mask_data)
    expected_mask_acq_next = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 1.0, 1.0, 1.0],
      [1.0, 0.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    self.assertTrue(torch.all(mask_acq_next == expected_mask_acq_next))
    change = torch.sum(mask_acq_next - mask_acq, dim=-1)
    self.assertTrue(torch.all(change == torch.tensor([0.0, 1.0, 1.0, 1.0])))
    mask_acq = mask_acq_next

    # Third acquisition.
    mask_acq_next = model.acquire(x, mask_acq, mask_data)
    expected_mask_acq_next = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 1.0, 1.0, 1.0],
    ])
    self.assertTrue(torch.all(mask_acq_next == expected_mask_acq_next))
    change = torch.sum(mask_acq_next - mask_acq, dim=-1)
    self.assertTrue(torch.all(change == torch.tensor([0.0, 1.0, 1.0, 1.0])))
    mask_acq = mask_acq_next

    # Fourth acquisition.
    mask_acq_next = model.acquire(x, mask_acq, mask_data)
    expected_mask_acq_next = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 1.0, 1.0, 1.0, 1.0],
    ])
    self.assertTrue(torch.all(mask_acq_next == expected_mask_acq_next))
    change = torch.sum(mask_acq_next - mask_acq, dim=-1)
    self.assertTrue(torch.all(change == torch.tensor([0.0, 1.0, 0.0, 1.0])))
    mask_acq = mask_acq_next

    # Fifth acquisition.
    mask_acq_next = model.acquire(x, mask_acq, mask_data)
    expected_mask_acq_next = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    self.assertTrue(torch.all(mask_acq_next == expected_mask_acq_next))
    change = torch.sum(mask_acq_next - mask_acq, dim=-1)
    self.assertTrue(torch.all(change == torch.tensor([0.0, 0.0, 0.0, 1.0])))
    mask_acq = mask_acq_next

    # Final acquisition.
    mask_acq_next = model.acquire(x, mask_acq, mask_data)
    expected_mask_acq_next = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    self.assertTrue(torch.all(mask_acq_next == expected_mask_acq_next))
    change = torch.sum(mask_acq_next - mask_acq, dim=-1)
    self.assertTrue(torch.all(change == torch.tensor([0.0, 0.0, 0.0, 0.0])))
    mask_acq = mask_acq_next

    self.assertTrue(torch.all(mask_acq == 1.0))

  def test_missing_data(self):
    config = {
      "num_con_features": 5,
      "num_cat_features": 0,
      "most_categories": 0,
      "out_dim": 10,
      "max_dim": None
    }
    model = TestModel(config)

    mask_data = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 0.0, 0.0, 1.0, 1.0],
      [1.0, 1.0, 0.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    mask_acq = torch.tensor([
      [1.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    x = torch.randn_like(mask_acq)


    # First acquisition.
    mask_acq = model.acquire(x, mask_acq, mask_data)
    mask_acq_true = torch.tensor([
      [1.0, 1.0, 0.0, 0.0, 1.0],
      [1.0, 0.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 1.0],
      [1.0, 0.0, 0.0, 0.0, 0.0]
    ])
    self.assertTrue(torch.all(mask_acq == mask_acq_true))

    # Second Acquisition.
    mask_acq = model.acquire(x, mask_acq, mask_data)
    mask_acq_true = torch.tensor([
      [1.0, 1.0, 0.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 1.0, 1.0],
      [1.0, 1.0, 0.0, 0.0, 0.0]
    ])
    self.assertTrue(torch.all(mask_acq == mask_acq_true))

    # Third Acquisition.
    mask_acq = model.acquire(x, mask_acq, mask_data)
    mask_acq_true = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 1.0, 0.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 0.0, 0.0]
    ])
    self.assertTrue(torch.all(mask_acq == mask_acq_true))

    # Fourth Acquisition.
    mask_acq = model.acquire(x, mask_acq, mask_data)
    mask_acq_true = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 0.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 0.0]
    ])
    self.assertTrue(torch.all(mask_acq == mask_acq_true))

    # Fifth Acquisition.
    mask_acq = model.acquire(x, mask_acq, mask_data)
    mask_acq_true = torch.tensor([
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0]
    ])
    self.assertTrue(torch.all(mask_acq == mask_acq_true))



if __name__ == "__main__":
  unittest.main()