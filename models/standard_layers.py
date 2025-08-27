"""Some standard layers for use in models. Such as fully connected.
We also include some input layers for different types of input, for the standard
models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.constants import min_sig


class MLP(nn.Module):
  """MLP with ReLU activations and batchnorm."""
  def __init__(self, in_dim, hidden_dim, out_dim, num_hidden):
    super().__init__()
    if num_hidden == 0:
      self.network = nn.Linear(in_dim, out_dim)
    else:
      self.network = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim)
      ]
      for _ in range(num_hidden-1):
        self.network.append(nn.Linear(hidden_dim, hidden_dim))
        self.network.append(nn.ReLU())
        self.network.append(nn.BatchNorm1d(hidden_dim))
      self.network.append(nn.Linear(hidden_dim, out_dim))
      self.network = nn.Sequential(*self.network)

  def forward(self, x):
    return self.network(x)


class StochasticMLP(nn.Module):
  """MLP that outputs a diagonal normal distribution."""
  def __init__(self, in_dim, hidden_dim, out_dim, num_hidden):
    super().__init__()
    self.network = MLP(in_dim, hidden_dim, 2*out_dim, num_hidden)
    self.out_dim = out_dim

  def forward(self, x):
    x = self.network(x)
    return x[:, :self.out_dim], F.softplus(x[:, self.out_dim:]) + min_sig


class ContinuousInput(nn.Module):
  """A layer that takes continuous inputs, [x*m , m] is the model input."""
  def __init__(self):
    super().__init__()

  def forward(self, x, mask):
    return torch.cat([x*mask, mask], dim=-1)


class CategoricalInput(nn.Module):
  """Input layer for categorical features.
  We add 1 to the categories and multiply by the mask, so that 0 is no feature
  and everything after that represents a true category. This is then turned
  into a one-hot encoding.
  """
  def __init__(self, most_categories):
    super().__init__()
    self.most_categories = most_categories

  def forward(self, x, mask):
    x = ((x + 1.0)*mask).long()
    return F.one_hot(x, num_classes=self.most_categories+1).float().view(x.shape[0], -1)


class MixedInput(nn.Module):
  """Input layer for mixed types, we strictly order so that all continuous
  are first, and then all categorical."""
  def __init__(self, num_con_features, most_categories):
    super().__init__()
    self.con_input = ContinuousInput()
    self.cat_input = CategoricalInput(most_categories)
    self.num_con_features = num_con_features

  def forward(self, x, mask):
    x_con = x[:, :self.num_con_features]
    mask_con = mask[:, :self.num_con_features]
    x_con = self.con_input(x_con, mask_con)

    x_cat = x[:, self.num_con_features:]
    mask_cat = mask[:, self.num_con_features:]
    x_cat = self.cat_input(x_cat, mask_cat)

    return torch.cat([x_con, x_cat], dim=-1)