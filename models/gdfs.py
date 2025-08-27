"""GDFS model implementation to be used in our codebase.
Paper: https://arxiv.org/abs/2301.00557
Code: https://github.com/iancovert/dynamic-selection/tree/main

NOTE: The temperature of the main training is very important, from the
published code the temperature progression is:
[0.2, 0.112, 0.063, 0.036, 0.02]
which does not do very much exploration early on, and can get stuck
in local minima quite easily. We found [2.0, 1.125, 0.632, 0.356, 0.2] worked
significantly better on the synthetic experiments, since there is earlier
exploration, so it can discover that feature 11 is crucial, even if not
immediately beneficial.

The majority of this code is adapted specifically from:
https://github.com/iancovert/dynamic-selection/blob/main/dynamic_selection/greedy.py
"""



import os.path as osp

import numpy as np

from sklearn.metrics import auc as sklearn_auc

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.distributions import RelaxedOneHotCategorical
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.base import BaseModel
from models.constants import lr_factor, min_lr, cooldown
from models.standard_layers import MLP



class GDFS(BaseModel):
  """GDFS model."""
  def __init__(self, config):
    super().__init__(config)
    self.pretrain_epochs = config["pretrain_epochs"]
    self.pretrain_lr = config["pretrain_lr"]
    self.temp_initial = config["temp_initial"]

    if config["share_parameters"]:
      to_hidden = nn.Sequential(
        MLP(
          in_dim=self.in_dim,
          hidden_dim=config["hidden_dim"],
          out_dim=config["hidden_dim"],
          num_hidden=int(config["num_hidden"]/2)
        ),
        nn.ReLU(),
        nn.BatchNorm1d(config["hidden_dim"]),
      )
      in_dim = config["hidden_dim"]
      num_hidden = config["num_hidden"]-int(config["num_hidden"]/2)
    else:
      to_hidden = nn.Identity()
      in_dim = self.in_dim
      num_hidden = config["num_hidden"]

    self.predictor_layers = nn.Sequential(
      to_hidden,
      MLP(
        in_dim=in_dim,
        hidden_dim=config["hidden_dim"],
        out_dim=self.out_dim,
        num_hidden=num_hidden,
      ),
    )
    self.selector_layers = nn.Sequential(
      to_hidden,
      MLP(
        in_dim=in_dim,
        hidden_dim=config["hidden_dim"],
        out_dim=self.num_features,
        num_hidden=num_hidden,
      ),
    )

  def predictor(self, x, mask):
    return self.predictor_layers(self.input_layer(x, mask))

  def selector(self, x, mask):
    return self.selector_layers(self.input_layer(x, mask))

  def predict(self, x, mask):
    return F.softmax(self.predictor(x, mask), dim=-1)

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    return F.softmax(self.selector(x, mask), dim=-1)

  def fit_parameters(self, train_data, val_data, save_path, metric_f):
    optimizer = Adam(self.parameters(), lr=self.pretrain_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=lr_factor,
                                  cooldown=cooldown, min_lr=min_lr, patience=self.patience)
    train_loader = DataLoader(train_data, batch_size=self.batchsize, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

    # Pretraining the predictor.
    for epoch in range(1, self.pretrain_epochs+1):
      self.train()
      epoch_loss = 0
      for x, y, m_data in train_loader:
        optimizer.zero_grad()
        x = x.to(self.device)
        y = y.to(self.device)
        m_data = m_data.to(self.device)
        loss = F.cross_entropy(self.predictor(x, self.subsample_mask(m_data)), y)
        loss.backward()
        epoch_loss += loss.item()/len(train_loader)
        optimizer.step()

      self.eval()
      val_metric, val_dict = self.calc_val_dict(val_loader, metric_f)
      scheduler.step(val_metric)
      if val_metric == scheduler.best:
        torch.save(self.state_dict(), osp.join(save_path, "best_model.pt"))

      # Print results of this epoch.
      print(f"\nPretraining Epoch: {epoch}/{self.pretrain_epochs}, Avg Loss: {epoch_loss:.3e}, ", end="")
      print(f"Val Metric: {val_metric:.3f}|{scheduler.best:.3f}", end="")
      for key, value in val_dict.items():
        print(f", {key}: {value:.3f}", end="")


    # Main training of the selection process.
    print("\n\nStarting main training")
    temp_progression = self.temp_initial*np.geomspace(1.0, 0.1, 5)
    best_hard_val_auc = -1

    for temp_id, temp in enumerate(temp_progression):
      # Each new temperature we load the model with the best "soft" performance
      # with soft sampling.
      print("")
      assert temp > 0.0, "Temperature must be > 0"
      self.load_state_dict(torch.load(osp.join(save_path, "best_model.pt")))
      optimizer = Adam(self.parameters(), lr=self.lr)
      scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=lr_factor,
                                    cooldown=cooldown, min_lr=min_lr, patience=self.patience)

      for epoch in range(1, self.epochs+1):
        self.train()
        epoch_loss = 0
        for x, y, m_data in train_loader:
          optimizer.zero_grad()
          x = x.to(self.device)
          y = y.to(self.device)
          m_data = m_data.to(self.device)
          m_acq = torch.zeros_like(m_data)

          for _ in range(self.max_dim):
            select_logits = self.selector(x, m_acq*m_data)
            m_av_sel = m_data*(1-m_acq)  # Mask for available and not selected features.
            select_logits = select_logits*m_av_sel - 1e6*(1-m_av_sel)
            m_soft = torch.max(m_acq, RelaxedOneHotCategorical(temp, logits=select_logits).rsample())

            loss = F.cross_entropy(self.predictor(x, m_soft*m_data), y)/self.max_dim
            loss.backward()
            epoch_loss += loss.item()/len(train_loader)

            # If nothing is available that we haven't already collected we want
            # to choose an arbitrary not selected feature.
            select_logits  += (1-m_acq)*1e6
            m_acq = torch.max(m_acq, F.one_hot(torch.argmax(select_logits, dim=-1), num_classes=self.num_features).float())

          optimizer.step()

        self.eval()
        val_auc = self.run_zero_acquisition(val_loader, metric_f)
        scheduler.step(val_auc)
        if val_auc > best_hard_val_auc:
          best_hard_val_auc = val_auc
          torch.save(self.state_dict(), osp.join(save_path, "best_model.pt"))

        # Print info about this epoch.
        print(f"Temp: {temp:.3e} ({temp_id+1}/{len(temp_progression)}), Epoch: {epoch}/{self.epochs}, ", end="")
        print(f"Val AUC: {val_auc:.3f}|{scheduler.best:.3f}|{best_hard_val_auc:.3f}, ", end="")
        print(f"Avg Loss: {epoch_loss:.3f}")

