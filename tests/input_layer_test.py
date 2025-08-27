"""Here we are testing the input layers. These are layers prep inputs 
for the standard layers.

We test by changing one value where the mask is 1 and seeing if only that part 
of the output changes.

We also change one value where the mask is 0 and see if there is no change, i.e.
when the mask is 0, the feature should provide no information, other than the
fact it is missing.

We also test defined outputs.
"""


import unittest

import torch

from models.standard_layers import ContinuousInput, CategoricalInput, MixedInput


# Constants.
batchsize = 5

# Continuous constants.
num_con_features = 8

# Categorical constants.
num_cat_features =  6
most_categories = 10


def produce_continuous_outputs(batchsize, num_features, mask_value):
  # Make outputs for a continuous input, the first value changes.
  continuous_input = ContinuousInput()
  x1 = torch.randn((batchsize, num_features))
  x2 = x1.clone()
  x2[:, 0] = -x2[:, 0]
  mask = torch.bernoulli(0.5*torch.ones((batchsize, num_features)))
  mask[:, 0] = mask_value
  in1 = continuous_input(x1, mask)
  in2 = continuous_input(x2, mask)
  return in1, in2


def produce_categorical_outputs(batchsize, num_features, most_categories, mask_value):
  # Make outputs for a categorical layer where the first value changes.
  categorical_input = CategoricalInput(most_categories)
  x1 = torch.randint(high=most_categories, size=(batchsize, num_features)).float()
  x2 = x1.clone()
  x2[:, 0] = (x2[:, 0] + 1.0) % most_categories
  mask = torch.bernoulli(0.5*torch.ones((batchsize, num_features)))
  mask[:, 0] = mask_value
  in1 = categorical_input(x1, mask)
  in2 = categorical_input(x2, mask)
  return in1, in2


# Continuous input.
class TestContinuousInput(unittest.TestCase):

  def test_mask1(self):
    in1, in2 = produce_continuous_outputs(batchsize, num_con_features, mask_value=1.0)
    self.assertFalse(torch.all(in1 == in2))

  def test_mask0(self):
    in1, in2 = produce_continuous_outputs(batchsize, num_con_features, mask_value=0.0)
    self.assertTrue(torch.all(in1 == in2))

  def test_values(self):
    con_input = ContinuousInput()
    x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    true_in = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 1.0], [0.0, 4.0, 0.0, 0.0, 1.0, 0.0]])
    test_in = con_input(x, mask)
    self.assertTrue(torch.all(test_in == true_in))


# Categorical input.
class TestCategoricalInput(unittest.TestCase):

  def test_mask1(self):
    in1, in2 = produce_categorical_outputs(batchsize, num_cat_features, most_categories, mask_value=1.0)
    self.assertFalse(torch.all(in1 == in2))

  def test_mask0(self):
    in1, in2 = produce_categorical_outputs(batchsize, num_cat_features, most_categories, mask_value=0.0)
    self.assertTrue(torch.all(in1 == in2))

  def test_values(self):
    most_categories = 3
    cat_input = CategoricalInput(most_categories)
    x = torch.tensor([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    true_in = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]])
    test_in = cat_input(x, mask)
    self.assertTrue(torch.all(test_in == true_in))


class TestMixedInput(unittest.TestCase):

  def test_values(self):
    num_con = 3
    most_categories = 3
    x_con = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    x_cat = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
    m_con = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    m_cat = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x = torch.cat([x_con, x_cat], dim=-1)
    mask = torch.cat([m_con, m_cat], dim=-1)
    true_in = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                             [0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    mixed_input = MixedInput(num_con, most_categories)
    test_in = mixed_input(x, mask)
    self.assertTrue(torch.all(test_in == true_in))


if __name__ == "__main__":
  unittest.main()