"""Test the data preprocessing functions, both standardization and CDF.
We test standardization with values we know the outcome of. We test CDF
by approximating the KL divergence of the output with the Standard
Normal distribution.
"""

import unittest

import numpy as np
import torch

from datasets.preprocessing_utils import Standardizer, VectorEmpiricalCDF


def normal(x):
  return np.exp(-x**2/2) / (2*np.pi)**0.5


def approx_kl_01(X):
  h_values, bin_edges = np.histogram(X, bins=20, density=True)
  bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
  kl = np.sum(normal(bin_centres)*(np.log(normal(bin_centres+1e-10)) - np.log(h_values+1e-10)))
  return np.abs(kl)


def generate_mog_data(num_samples1, num_features):
  x1 = -15.0 + 6.0*torch.randn((num_samples1, num_features))
  x2 = 10.0 + 0.5*torch.randn((int(0.5*num_samples1), num_features))
  x3 = -3.0 + 1.0*torch.randn((int(0.8*num_samples1), num_features))
  x4 = 5.0 + 2.0*torch.randn((int(1.6*num_samples1), num_features))
  return torch.cat([x1, x2, x3, x4], dim=0)


class TestStandardizer(unittest.TestCase):

  def test_no_mask(self):
    X = torch.tensor([[1.3, 2.1, 3.9], [-2.4, 3.0, 10.0], [-1.0, 0.9, 5.6]])
    standardizer = Standardizer()
    standardizer.fit(X)
    true_mean = torch.tensor([[-0.7, 2.0, 6.5]])
    # Sample standard deviation correction with Bessel's correction.
    true_std = torch.tensor([[(3.49)**0.5, (1.11)**0.5, (9.91)**0.5]])
    self.assertTrue(torch.allclose(standardizer.mean, true_mean))
    self.assertTrue(torch.allclose(standardizer.std, true_std))

    # Check the transformation is correct.
    test_point = torch.tensor([[3.5, -1.0, 6.0]])
    test_point_standardized = standardizer.transform(test_point)
    true_standardized = torch.tensor([[2.248208456, -2.847473987, -0.1588302345]])
    self.assertTrue(torch.allclose(test_point_standardized, true_standardized))

  def test_mask(self):
    X = torch.tensor([
      [1.3, -8.0, 13.0],
      [-2.4, 3.0, 10.0],
      [-1.0, 0.9, 11.4],
      [20.0, -50, 5.6],
      [100.0, 100.0, 3.9],
      [0.0, 2.1, 5.4]
    ])
    mask = torch.tensor([
      [1.0, 0.0, 0.0],
      [1.0, 1.0, 1.0],
      [1.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
      [0.0, 0.0, 1.0],
      [0.0, 1.0, 0.0]
    ])
    standardizer = Standardizer()
    standardizer.fit(X, mask)
    true_mean = torch.tensor([[-0.7, 2.0, 6.5]])
    # Sample standard deviation correction with Bessel's correction.
    true_std = torch.tensor([[(3.49)**0.5, (1.11)**0.5, (9.91)**0.5]])
    self.assertTrue(torch.allclose(standardizer.mean, true_mean))
    self.assertTrue(torch.allclose(standardizer.std, true_std))

    test_point = torch.tensor([
      [3.5, 4.0, 5.4],
      [11.0, -1.0, 5.0],
      [9.0, -1.0, 6.0]
      ])
    test_mask = torch.tensor([
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ])
    true_standardized = torch.tensor([
      [2.248208456, 0.0, 0.0],
      [0.0, -2.847473987, 0.0],
      [0.0, 0.0, -0.1588302345]
    ])
    test_point_standardized = standardizer.transform(test_point, test_mask)
    self.assertTrue(torch.allclose(test_point_standardized, true_standardized))


class TestVectorEmpiricalCDF(unittest.TestCase):

  def test_no_mask(self):
    # Create data from an aribtrary distribution, check after transform
    # it matches standard normal. Distribution is 4 mode Mixture of Gaussians.
    X = generate_mog_data(20_000, 4)
    empirical_cdf = VectorEmpiricalCDF(num_bins=1000, size_normal=0.0, ratio_uniform=0.0)
    empirical_cdf.fit(X)
    X_cdf = empirical_cdf.transform(X)
    X_cdf = X_cdf.numpy()
    for i in range(4):
      self.assertTrue(approx_kl_01(X_cdf[:, i]) < 0.001)

    # Test data from the same distribution.
    X_test = generate_mog_data(20_000, 4)
    X_cdf = empirical_cdf.transform(X_test)
    X_cdf = X_cdf.numpy()
    for i in range(4):
      self.assertTrue(approx_kl_01(X_cdf[:, i]) < 0.001)

    # Test data from a different distribution.
    X_test = torch.randn((20_000, 4))
    X_cdf = empirical_cdf.transform(X_test)
    X_cdf = X_cdf.numpy()
    for i in range(4):
      self.assertTrue(approx_kl_01(X_cdf[:, i]) > 2.0)

  def test_defined_cdf(self):
    # Define the CDF to be:
    # y = 0.1x for 0<=x<=1
    # y = 0.8x-0.7 for 1<=x<=2
    # y = 0.1x + 0.7 for 2<=x<=3
    num_cdf1 = 100_000
    cdf1 = (0.1-0.0)*torch.rand((num_cdf1, 1)) + 0.0
    cdf2 = (0.9-0.1)*torch.rand((8*num_cdf1, 1)) + 0.1
    cdf3 = (1-0.9)*torch.rand((num_cdf1, 1)) + 0.9
    x1 = (cdf1 + 0.0) / 0.1
    x2 = (cdf2 + 0.7) / 0.8
    x3 = (cdf3 - 0.7) / 0.1
    X = torch.cat([x1, x2, x3], dim=0)

    empirical_cdf = VectorEmpiricalCDF(num_bins=100, size_normal=0.0, ratio_uniform=0.0)
    empirical_cdf.fit(X)

    test_x1 = (1.0 - 0.0)*torch.rand((50, 1)) + 0.0
    test_x2 = (2.0 - 1.0)*torch.rand((50, 1)) + 1.0
    test_x3 = (3.0 - 2.0)*torch.rand((50, 1)) + 2.0

    test_x = torch.cat([test_x1, test_x2, test_x3], dim=0)
    test_cdf = empirical_cdf.empirical_cdf(test_x)

    true_cdf1 = 0.1*test_x1
    true_cdf2 = 0.8*test_x2 - 0.7
    true_cdf3 = 0.1*test_x3 + 0.7
    true_cdf = torch.cat([true_cdf1, true_cdf2, true_cdf3], dim=0)
    self.assertTrue(torch.mean(torch.abs(test_cdf - true_cdf)).item() < 0.001)
    self.assertTrue(torch.max(torch.abs(test_cdf - true_cdf)) < 0.005)

  def test_defined_cdf_perfect_fit(self):
    # Test with the known cdf, however we manually set the parameters of the
    # empirical cdf to be the same as the true cdf. This tests the ability
    # to search sorted and gather correctly.
    empirical_cdf = VectorEmpiricalCDF(num_bins=100, size_normal=0.0, ratio_uniform=0.0)
    empirical_cdf.cdf = torch.tensor([[0.0, 0.1, 0.9, 1.0]])
    empirical_cdf.s = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    empirical_cdf.m = torch.tensor([[0.0, 0.1, 0.8, 0.1, 0.0]])

    test_points = torch.tensor([[-1.0], [-0.1], [0.0], [0.1], [0.6], [1.0], 
                                [1.2], [1.5], [1.9], [2.0], [2.5], [3.0], 
                                [3.5], [4.0]])
    true_cdf = torch.tensor([[0.0], [0.0], [0.0], [0.01], [0.06], [0.1],
                             [0.26], [0.5], [0.82], [0.9], [0.95], [1.0],
                             [1.0], [1.0]])
    test_cdf = empirical_cdf.empirical_cdf(test_points)
    self.assertTrue(torch.allclose(test_cdf, true_cdf))

  def test_mask(self):
    # Create data from an aribtrary distribution, check after transform
    # it matches standard normal. Distribution is 4 mode Mixture of Gaussians.
    X_true = generate_mog_data(20000, 4)
    X_false = torch.randn((5000, 4))
    M_true = torch.ones_like(X_true)
    M_false = torch.zeros_like(X_false)

    X = torch.cat([X_true, X_false], dim=0)
    M = torch.cat([M_true, M_false], dim=0)

    # Shuffle the indices for each feature.
    for i in range(4):
      shuffle_ids = torch.randperm(X.shape[0])
      X[:, i] = X[shuffle_ids, i]
      M[:, i] = M[shuffle_ids, i]

    empirical_cdf = VectorEmpiricalCDF(num_bins=1000, size_normal=0.0, ratio_uniform=0.0)
    empirical_cdf.fit(X, M)

    # Test real data gives correct histogram.
    X_cdf = empirical_cdf.transform(X_true, M_true)
    X_cdf = X_cdf.numpy()
    for i in range(4):
      self.assertTrue(approx_kl_01(X_cdf[:, i]) < 0.001)

    # Test data from the same distribution gives correct histogram.
    X_test = generate_mog_data(20000, 4)
    X_cdf = empirical_cdf.transform(X_test)
    X_cdf = X_cdf.numpy()
    for i in range(4):
      self.assertTrue(approx_kl_01(X_cdf[:, i]) < 0.001)

    # Test Fake data gives zeros.
    X_cdf = empirical_cdf.transform(X_false, M_false)
    self.assertTrue(torch.all(X_cdf==torch.zeros_like(X_cdf)))

    # Test data from a different distribution.
    X_test = torch.randn((20_000, 4))
    X_cdf = empirical_cdf.transform(X_test)
    X_cdf = X_cdf.numpy()
    for i in range(4):
      self.assertTrue(approx_kl_01(X_cdf[:, i]) > 2.0)

  def test_linspace(self):
    empirical_cdf = VectorEmpiricalCDF()
    num_features = 20
    steps = 10
    x_0 = torch.rand((num_features))
    x_1 = torch.rand((num_features)) + 4.0
    test_linspace = empirical_cdf.linspace_batched(x_0, x_1, steps)
    looped_linspace = torch.empty((num_features, steps))
    for i in range(num_features):
      looped_linspace[i] = torch.linspace(x_0[i], x_1[i], steps)
    self.assertTrue(torch.allclose(test_linspace, looped_linspace))



if __name__ == "__main__":
  np.random.seed(42)
  torch.manual_seed(42)
  unittest.main()