"""Test the block layers, including the BlockLinear and BlockStochasticMLPs."""

import unittest

import torch

from models.sefa import BlockLinear, ContinuousBlockStochasticMLP, CategoricalBlockStochasticMLP


class TestBlockLinear(unittest.TestCase):
  def test_correct_values(self):
    # Test if one layer produces same output as a slow but correct loop.
    in_dim = 4
    out_dim = 7
    num_blocks = 8
    batchsize = 5
    model = BlockLinear(in_dims=in_dim, out_dims=out_dim, num_blocks=num_blocks)
    x = torch.randn(batchsize, in_dim*num_blocks)
    weight = model.weight
    bias = model.bias
    model_out = model(x)
    loop_out = torch.zeros(batchsize, out_dim*num_blocks)
    for batch in range(batchsize):
      for block in range(num_blocks):
        loop_out[batch, block*out_dim:(block+1)*out_dim] = torch.matmul(weight[block], x[batch, block*in_dim:(block+1)*in_dim]) + bias[block*out_dim:(block+1)*out_dim]
    self.assertTrue(torch.allclose(model_out, loop_out, rtol=1e-4, atol=1e-7))
  
  def test_independence(self):
    # Test if the model produces the same output if we change the input of other
    # blocks.
    in_dim = 4
    out_dim = 7
    num_blocks = 8
    batchsize = 5
    model = BlockLinear(in_dims=in_dim, out_dims=out_dim, num_blocks=num_blocks)
    in1 = torch.randn(batchsize, in_dim*num_blocks)
    in2 = torch.randn(batchsize, in_dim*num_blocks)
    in2[:, :in_dim] = in1[:, :in_dim]
    out1 = model(in1)
    out2 = model(in2)
    # Check first block is the same since it uses the same data.
    self.assertTrue(torch.all(out1[:, :out_dim] == out2[:, :out_dim]))
    # Check output is not the same.
    self.assertFalse(torch.all(out1[:, out_dim:] == out2[:, out_dim:]))


class TestContinuousBlockStochasticMLP(unittest.TestCase):
  def test_mask0(self):
    # Set mask value of first feature to 0. See if its output doesn't change when
    # all others do.
    for num_hidden in [0, 1, 5]:
      batchsize = 5
      num_con_features = 4
      hidden_dim = 100
      out_dim = 7
      x = torch.randn((batchsize, num_con_features))
      mask = torch.bernoulli(0.5*torch.ones_like(x))
      mask[:, 0] = 0.0
      mlp = ContinuousBlockStochasticMLP(num_con_features, hidden_dim, out_dim, num_hidden)
      mlp.eval()
      mu1, sig1 = mlp(x, mask)
      x = torch.randn((batchsize, num_con_features))
      mu2, sig2 = mlp(x, mask)
      # Check first output matches, since it is masked out.
      self.assertTrue(torch.all(mu1[:, :out_dim] == mu2[:, :out_dim]))
      self.assertTrue(torch.all(sig1[:, :out_dim] == sig2[:, :out_dim]))
      # Check none of the other outputs don't match.
      self.assertFalse(torch.all(mu1[:, out_dim:] == mu2[:, out_dim:]))
      self.assertFalse(torch.all(sig1[:, out_dim:] == sig2[:, out_dim:]))

  def test_mask1(self):
    # Set mask value to 1. See if the value changes, but no others do.
    for num_hidden in [0, 1, 5]:
      batchsize = 5
      num_con_features = 4
      hidden_dim = 100
      out_dim = 7
      x = torch.randn((batchsize, num_con_features))
      mask = torch.bernoulli(0.5*torch.ones_like(x))
      mask[:, 0] = 1.0
      mlp = ContinuousBlockStochasticMLP(num_con_features, hidden_dim, out_dim, num_hidden)
      mlp.eval()
      mu1, sig1 = mlp(x, mask)
      x[:, 0] = torch.randn((batchsize,))
      mu2, sig2 = mlp(x, mask)
      # Check the dims after first match.
      self.assertTrue(torch.all(mu1[:, out_dim:] == mu2[:, out_dim:]))
      self.assertTrue(torch.all(sig1[:, out_dim:] == sig2[:, out_dim:]))
      # Check the change in first dim, with mask=1.0 causes change to output.
      self.assertFalse(torch.all(mu1[:, :out_dim] == mu2[:, :out_dim]))
      self.assertFalse(torch.all(sig1[:, :out_dim] == sig2[:, :out_dim]))


class TestCategoricalBlockStochasticMLP(unittest.TestCase):
  def test_true_value(self):
    # Test if the model produces the correct output first.
    num_cat_features = 4
    out_dims = 7
    most_categories = 3
    mlp = CategoricalBlockStochasticMLP(num_cat_features, out_dims, most_categories)
    mlp.eval()
    x = torch.tensor([
      [1.0, 0.0, 0.0, 2.0],
      [0.0, 1.0, 2.0, 0.0],
      [1.0, 2.0, 0.0, 0.0],
    ])
    mask = torch.tensor([
      [0.0, 0.0, 1.0, 1.0],
      [0.0, 1.0, 1.0, 1.0],
      [1.0, 0.0, 1.0, 0.0]
    ])
    mu, _ = mlp(x, mask)
    for batch in range(x.shape[0]):
      for feature in range(x.shape[1]):
        if mask[batch, feature] != 0.0:
          true_out = mlp.mu_embeddings.weight[feature*(most_categories+1):(feature+1)*(most_categories+1)][(x[batch, feature] + 1).long()]
        else:
          true_out = mlp.mu_embeddings.weight[feature*(most_categories+1):(feature+1)*(most_categories+1)][0]
        self.assertTrue(torch.allclose(mu[batch, feature*out_dims:(feature+1)*out_dims], true_out))

  def test_mask0(self):
    # Set mask value of first feature to 0. See if it's output doesn't change when
    # all others do.
    batchsize = 5
    num_cat_features = 4
    out_dim = 7
    most_categories = 3
    x = torch.randint(high=most_categories, size=(batchsize, num_cat_features)).float()
    mask = torch.bernoulli(0.5*torch.ones_like(x))
    mask[:, 0] = 0.0
    mlp = CategoricalBlockStochasticMLP(num_cat_features, out_dim, most_categories)
    mlp.eval()
    mu1, sig1 = mlp(x, mask)
    x = torch.randint(high=most_categories, size=(batchsize, num_cat_features)).float()
    mu2, sig2 = mlp(x, mask)
    self.assertTrue(torch.all(mu1[:, :out_dim] == mu2[:, :out_dim]))
    self.assertTrue(torch.all(sig1[:, :out_dim] == sig2[:, :out_dim]))
    # Check none are the same.
    self.assertFalse(torch.all(mu1[:, out_dim:] == mu2[:, out_dim:]))
    self.assertFalse(torch.all(sig1[:, out_dim:] == sig2[:, out_dim:]))

  def test_mask1(self):
    # Set mask value to 1. See if the value changes, but no others do.
    batchsize = 5
    num_cat_features = 4
    out_dim = 7
    most_categories = 3
    x = torch.randint(high=most_categories, size=(batchsize, num_cat_features)).float()
    mask = torch.bernoulli(0.5*torch.ones_like(x))
    mask[:, 0] = 1.0
    mlp = CategoricalBlockStochasticMLP(num_cat_features, out_dim, most_categories)
    mlp.eval()
    mu1, sig1 = mlp(x, mask)
    x[:, 0] = torch.randint(high=most_categories, size=(batchsize,)).float()
    mu2, sig2 = mlp(x, mask)
    self.assertTrue(torch.all(mu1[:, out_dim:] == mu2[:, out_dim:]))
    self.assertTrue(torch.all(sig1[:, out_dim:] == sig2[:, out_dim:]))
    # Check none are the same.
    self.assertFalse(torch.all(mu1[:, :out_dim] == mu2[:, :out_dim]))
    self.assertFalse(torch.all(sig1[:, :out_dim] == sig2[:, :out_dim]))


if __name__ == "__main__":
  unittest.main()