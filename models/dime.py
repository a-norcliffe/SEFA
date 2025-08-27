"""DIME from the ICLR 2024 paper: https://arxiv.org/abs/2306.03301
Code: https://github.com/suinleelab/DIME/tree/main

The idea is that we have a predictive model, and a value model. And like
in Mutual Neural Estimation the value model can be used to estimate the mutual
information directly.

Most code based on: https://github.com/suinleelab/DIME/blob/main/dime/cmi_estimator.py
A lot of code copied from our GDFS implementation.
"""

import os.path as osp

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.base import BaseModel
from models.constants import log_eps, lr_factor, min_lr, cooldown
from models.standard_layers import MLP


class DIME(BaseModel):
  """DIME implementation."""
  def __init__(self, config):
    super().__init__(config)
    self.pretrain_epochs = config["pretrain_epochs"]
    self.pretrain_lr = config["pretrain_lr"]
    self.eps_initial = config["eps_initial"]

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
      )
    )
    self.value_layers = nn.Sequential(
      to_hidden,
      MLP(
        in_dim=in_dim,
        hidden_dim=config["hidden_dim"],
        out_dim=self.num_features,
        num_hidden=num_hidden,
      ),
      nn.Sigmoid(),
    )

  def predictor(self, x, mask):
    return self.predictor_layers(self.input_layer(x, mask))

  def predict(self, x, mask):
    return F.softmax(self.predictor(x, mask), dim=-1)

  def value(self, x, mask):
    # Note this is unscaled. But scaling is included in the calculate 
    # acquisition scores and during training loop.
    return self.value_layers(self.input_layer(x, mask))

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    H = self.predict(x, mask)
    H = -torch.sum(H*torch.log(H + log_eps), dim=-1, keepdim=True)
    return H*self.value(x, mask)

  def fit_parameters(self, train_data, val_data, save_path, metric_f):
    # Pretraining the predictor.
    optimizer = Adam(self.parameters(), lr=self.pretrain_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=lr_factor,
                                  cooldown=cooldown, min_lr=min_lr, patience=self.patience)
    train_loader = DataLoader(train_data, batch_size=self.batchsize, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

    print(f"Starting Pretraining")
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


    # Set up the epsilon progression for main training
    print("\n\nStarting main training")
    eps_progression = self.eps_initial*np.array([1.0, 0.25, 0.05, 0.005])
    best_hard_val_auc = -1

    for eps_id, eps in enumerate(eps_progression):
      # Each new epsilon we load best version of model and set up new optimizer
      # and learning rate scheduler.
      print("")
      assert eps >= 0, "Epsilon must be positive."
      assert eps <= 1, "Epsilon must be less than 1."
      self.load_state_dict(torch.load(osp.join(save_path, "best_model.pt")))
      optimizer = Adam(self.parameters(), lr=self.lr)
      scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=lr_factor,
                                    cooldown=cooldown, min_lr=min_lr, patience=self.patience)

      for epoch in range(1, self.epochs+1):
        self.train()
        epoch_pred_loss = 0
        epoch_value_loss = 0
        for x, y, m_data in train_loader:
          optimizer.zero_grad()
          x = x.to(self.device)
          y = y.to(self.device)
          m_data = m_data.to(self.device)
          m_acq = torch.zeros_like(m_data)

          p_prev = self.predictor(x, m_acq*m_data)
          loss_prev = F.cross_entropy(p_prev, y, reduction="none")
          loss_tmp = torch.mean(loss_prev)/(self.max_dim+1)
          loss_tmp.backward()
          epoch_pred_loss += loss_tmp.item()/len(train_loader)
          loss_prev = loss_prev.detach()

          for _ in range(self.max_dim):
            p_prev = F.softmax(p_prev, dim=-1).detach()
            H_prev = -torch.sum(p_prev*torch.log(p_prev + log_eps), dim=-1, keepdim=True).detach()
            cmi = H_prev*self.value(x, m_acq*m_data)
            cmi = cmi*(1-m_acq)*m_data + (1-m_acq)*1e-6

            #Make an acquisition either uniformly or with the scores.
            cmi_max = F.one_hot(torch.argmax(cmi, dim=-1), num_classes=self.num_features).float()

            unif = torch.rand_like(cmi)*(1-m_acq)*m_data + (1-m_acq)*1e-6
            unif = F.one_hot(torch.argmax(unif, dim=-1), num_classes=self.num_features).float()

            c_or_u = torch.bernoulli(torch.full_like(cmi[:, 0:1], 1-eps))
            m_update = c_or_u*cmi_max + (1-c_or_u)*unif
            m_acq = torch.max(m_acq, m_update)

            # Get loss with newly acquired feature.
            p_next = self.predictor(x, m_acq*m_data)
            loss_next = F.cross_entropy(p_next, y, reduction="none")
            loss_tmp = torch.mean(loss_next)/(self.max_dim+1)
            loss_tmp.backward()
            epoch_pred_loss += loss_tmp.item()/len(train_loader)
            loss_next = loss_next.detach()

            # Calculate change in loss, and whether the CMI network predicted that.
            delta = (loss_prev - loss_next).detach()
            cmi_preds = torch.sum(cmi*m_update, dim=-1)
            cmi_loss_tmp = torch.mean((cmi_preds - delta)**2)/self.max_dim
            cmi_loss_tmp.backward()
            epoch_value_loss += cmi_loss_tmp.item()/len(train_loader)
            loss_prev = loss_next

          optimizer.step()

        self.eval()
        val_auc = self.run_zero_acquisition(val_loader, metric_f)
        scheduler.step(val_auc)

        if val_auc > best_hard_val_auc:
          best_hard_val_auc = val_auc
          torch.save(self.state_dict(), osp.join(save_path, "best_model.pt"))

        print(f"Eps: {eps:.3e} ({eps_id+1}/{len(eps_progression)}), Epoch: {epoch}/{self.epochs}, ", end="")
        print(f"Val AUC: {val_auc:.3f}|{scheduler.best:.3f}|{best_hard_val_auc:.3f}, ", end="")
        print(f"Pred Loss: {epoch_pred_loss:.3f}, CMI Loss: {epoch_value_loss:.3e}")
