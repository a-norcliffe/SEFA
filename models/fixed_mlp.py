"""Standard MLP model, acquires based on a fixed order."""

import os.path as osp

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.base import BaseModel
from models.standard_layers import MLP



class FixedMLP(BaseModel):
  """Simple MLP model. Acquires based on a fixed global ordering."""
  def __init__(self, config):
    super().__init__(config)
    self.predictor = MLP(
      in_dim=self.in_dim,
      hidden_dim=config["hidden_dim"],
      out_dim=self.out_dim,
      num_hidden=config["num_hidden"],
    )

  def load(self, path):
    self.load_state_dict(torch.load(osp.join(path, "best_model.pt"), map_location=self.device))
    self.fixed_order_scores = torch.load(osp.join(path, "fixed_order_scores.pt"), map_location=self.device)
    self.eval()

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    return self.fixed_order_scores.repeat(mask.shape[0], 1)

  def forward(self, x, mask):
    return self.predictor(self.input_layer(x, mask))

  def predict(self, x, mask):
    # Predict has to give the distribution, not logits.
    return F.softmax(self.forward(x, mask), dim=-1)

  def loss_func(self, x, y, mask, data_mask=None):
    return F.cross_entropy(self.forward(x, mask), y)

  def post_fitting_tasks(self, train_data, val_data, save_path, metric_f):
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    self.fixed_order_scores = self.find_fixed_order(train_loader, metric_f)
    torch.save(self.fixed_order_scores, osp.join(save_path, "fixed_order_scores.pt"))

  @torch.no_grad()
  def find_fixed_order(self, train_loader, metric_func):
    # Note this can be a slow process, it is best to do with as large a
    # batchsize as possible.
    print("\n\nFinding greedy fixed ordering")
    possible_features = list(range(self.num_features))
    fixed_order = []
    while len(possible_features) > 1:
      print(f"Remaining features: {len(possible_features)}/{self.num_features}")
      best_metric = -1
      best_feature = -1
      for feature in possible_features:
        avg_metric = 0.0
        # Calculate the metric for each feature.
        for x, y, m_data in train_loader:
          x = x.to(self.device)
          y = y.to(self.device)
          m_data = m_data.to(self.device)
          m_tmp = torch.zeros_like(m_data)
          m_tmp[:, np.array(fixed_order + [feature]).astype(int)] = 1.0
          m_tmp *= m_data
          avg_metric += metric_func(self.predict(x, m_tmp), y)/len(train_loader)
        if avg_metric > best_metric:
          best_feature = int(feature)
          best_metric = avg_metric
      # Add to the fixed order.
      fixed_order.append(best_feature)
      possible_features.remove(best_feature)
    # Add final feature.
    fixed_order.append(possible_features[0])
    # Convert the order to scores.
    scores = torch.zeros(self.num_features, device=self.device, dtype=torch.float32)
    for i in range(self.num_features):
      scores[fixed_order[i]] = self.num_features - i
    return scores
