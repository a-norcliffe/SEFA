"""Random ordering, that uses an MLP for prediction."""

import os.path as osp

import torch

from models.fixed_mlp import FixedMLP


class RandomOrdering(FixedMLP):
  def __init__(self, config):
    super().__init__(config)

  def load(self, path):
    self.load_state_dict(torch.load(osp.join(path, "best_model.pt"), map_location=self.device))
    self.eval()

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    return torch.rand_like(mask)