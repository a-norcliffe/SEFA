"""Test the ACFlow modules are fully invertible. Since they use the
conditioning. Needs to be in eval mode. Also test the trainable prior.
"""

import os

import unittest

import numpy as np
import torch

from models.acflow import RealNVPLayer, FlowNetwork, PriorNetwork, ACFlow


class TestComponents(unittest.TestCase):

  def test_one_layer(self):
    batchsize = 100
    num_features = 50
    num_conditioning_features=100
    hidden_dim=100
    num_hidden=3

    layer = RealNVPLayer(
      num_features=num_features,
      num_conditioning_features=num_conditioning_features,
      hidden_dim=hidden_dim,
      num_hidden=num_hidden
    )
    layer.eval()
    with torch.no_grad():
      x = torch.randn((batchsize, num_features))
      conditioning = torch.randn((batchsize, num_conditioning_features))
      y, _ = layer(x, conditioning)
      x_inv, _ = layer.inverse(y, conditioning)
    self.assertTrue(torch.allclose(x, x_inv))  # Check inversion produces same.
    self.assertFalse(torch.allclose(x, y, atol=0.1))  # Check forward produces something different.

  def test_block(self):
    batchsize = 100
    num_features = 50
    num_conditioning_features=100
    hidden_dim=100
    num_hidden=3
    num_flow_modules=5

    flow_model = FlowNetwork(
      num_features=num_features,
      num_conditioning_features=num_conditioning_features,
      hidden_dim=hidden_dim,
      num_hidden=num_hidden,
      num_flow_modules=num_flow_modules,
    )
    flow_model.eval()

    with torch.no_grad():
      x = torch.randn((batchsize, num_features))
      conditioning = torch.randn((batchsize, num_conditioning_features))
      y, _ = flow_model(x, conditioning)
      x_inv, _ = flow_model.inverse(y, conditioning)
    self.assertTrue(torch.allclose(x, x_inv, atol=1e-6))  # Check inversion produces same.
    self.assertFalse(torch.allclose(x, y, atol=0.1))  # Check forward produces something different.

  def test_prior_shapes(self):
    batchsize = 100
    num_features = 50
    num_conditioning_features = 100
    hidden_dim = 100
    num_hidden = 3
    num_samples = 300

    prior = PriorNetwork(
      num_features=num_features,
      num_conditioning_features=num_conditioning_features,
      hidden_dim=hidden_dim,
      num_hidden=num_hidden,
    )
    prior.eval()

    with torch.no_grad():
      conditioning = torch.randn((batchsize, num_conditioning_features))
      samples = prior.conditional_latent_sample(conditioning, num_samples)
      assert samples.shape == (batchsize*num_samples, num_features)

      z = torch.randn((batchsize, num_features))
      log_p = prior.log_likelihood(z, conditioning)
      assert log_p.shape == (batchsize,)



dataset_dict = {
  "num_con_features": 5,
  "num_cat_features": 5,
  "most_categories": 3,
  "out_dim": 2,
  "metric": "auroc",
  "max_dim": None,
  "log_class_probs": torch.tensor([np.log(0.5), np.log(0.5)]),
}
config = {
  "hidden_dim_prior": 120,
  "num_hidden_prior": 1,
  "num_flow_modules": 4,
  "hidden_dim_flow": 120,
  "num_hidden_flow": 1,
  "lambda_nll": 0.1,
  "lambda_xent_sub": 1.0,
  "lambda_xent_full": 1.0,
  "num_samples_acquire": 50,
  "epochs": 1,
  "lr": 0.001,
  "batchsize": 128,
  "patience": 3,
}
for k, v in dataset_dict.items():
  config[k] = v

class TestACFlow(unittest.TestCase):
  def test_initialisation(self):
    model = ACFlow(config)

  def test_prediction(self):
    model = ACFlow(config)
    model.eval()
    batchsize = 20
    x = torch.randn((batchsize, config["num_con_features"] + config["num_cat_features"]))
    m = torch.bernoulli(0.5*torch.ones_like(x))
    py = model.predict(x, m)
    assert py.shape == (batchsize, config["out_dim"])

  def test_sample(self):
    model = ACFlow(config)
    model.eval()
    batchsize = 20
    x = torch.randn((batchsize, config["num_con_features"] + config["num_cat_features"]))
    m = torch.bernoulli(0.5*torch.ones_like(x))
    x_samples = model.unconditional_samples(x, m, m, num_samples=10)
    assert x_samples.shape == (batchsize*10, config["num_con_features"] + config["num_cat_features"])

  def test_loading(self):
    # The model can have issues when initialising, because there is randomness
    # in initialisation. So we want to check loading works and produces
    # same model.
    model1 = ACFlow(config)
    model1.eval()

    batchsize = 20
    x = torch.randn((batchsize, config["num_con_features"] + config["num_cat_features"]))
    m = torch.bernoulli(0.5*torch.ones_like(x))
    py1 = model1.predict(x, m)
    torch.manual_seed(42)
    x_samples1 = model1.unconditional_samples(x, m, m, num_samples=10)

    torch.save(model1.state_dict(), "best_model.pt")

    model2 = ACFlow(config)
    model2.eval()
    model2.load("")
    py2 = model2.predict(x, m)
    torch.manual_seed(42)
    x_samples2 = model2.unconditional_samples(x, m, m, num_samples=10)

    assert torch.all(py1 == py2)
    assert torch.all(x_samples1 == x_samples2)

    # Remove the state dict from files
    os.remove("best_model.pt")

  def test_masking0(self):
    model = ACFlow(config)
    model.eval()
    batchsize = 20
    x = torch.randn((batchsize, config["num_con_features"] + config["num_cat_features"]))
    m = torch.bernoulli(0.5*torch.ones_like(x))
    m[:, 0] = 0.0
    py1 = model.predict(x, m)

    x[:, 0] = torch.randn((batchsize))
    py2 = model.predict(x, m)
    assert torch.all(py1 == py2)

  def test_masking1(self):
    model = ACFlow(config)
    model.eval()
    batchsize = 20
    x = torch.randn((batchsize, config["num_con_features"] + config["num_cat_features"]))
    m = torch.bernoulli(0.5*torch.ones_like(x))
    m[:, 0] = 1.0
    py1 = model.predict(x, m)

    x[:, 0] = torch.randn((batchsize))
    py2 = model.predict(x, m)

    self.assertFalse(torch.allclose(py1, py2, atol=0.1))



if __name__ == "__main__":
  unittest.main()