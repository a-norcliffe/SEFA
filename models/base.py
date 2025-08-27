"""Base class for any acquisition model.

All models inherit from this class. This class is responsible for the main
training loop, saving/loading, and acquisition. We acquire by scoring each
feature based on the information we have, each model has its own scoring function.
Then we choose the feature with the largest score that is available and that
we have not already acquired.
"""

import os.path as osp
import gc

import numpy as np

from sklearn.metrics import auc as sklearn_auc

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.constants import lr_factor, cooldown, min_lr, acquisition_batch_limit
from models.standard_layers import ContinuousInput, CategoricalInput, MixedInput



class BaseModel(nn.Module):
  """Base acquisition model. Is able to predict y and update the mask.
  Models change how they predict and how they score feature acquisitions.
  Model parameter fitting can also change based on the specific model.
  """
  def __init__(self, config):
    super().__init__()
    # Input parameters.
    self.in_dim = 2*config["num_con_features"] + config["num_cat_features"]*(config["most_categories"]+1)
    if config["num_con_features"] == 0:
      self.input_type = "categorical"
      self.input_layer = CategoricalInput(config["most_categories"])
    elif config["num_cat_features"] == 0:
      self.input_type = "continuous"
      self.input_layer = ContinuousInput()
    elif config["num_con_features"] > 0 and config["num_cat_features"] > 0:
      self.input_type = "mixed"
      self.input_layer = MixedInput(config["num_con_features"], config["most_categories"])
    else:
      raise ValueError("Invalid input configuration, must have at least one continuous or categorical feature.")
    # Optimization parameters, some models (like RL) do not use these.
    self.epochs = config["epochs"] if "epochs" in config else None
    self.lr = config["lr"] if "lr" in config else None
    self.batchsize = config["batchsize"] if "batchsize" in config else None
    self.patience = config["patience"] if "patience" in config else None
    # Other parameters.
    self.num_features = config["num_con_features"] + config["num_cat_features"]
    self.max_dim = config["max_dim"] if ("max_dim" in config and config["max_dim"] is not None) else self.num_features
    self.out_dim = config["out_dim"]
    self.batch_limit = acquisition_batch_limit

  @property
  def device(self):
    return next(self.parameters()).device

  def clear_cache(self):
    if self.device != torch.device("cpu"):
      torch.cuda.empty_cache()
    gc.collect()

  def predict(self, x, mask):
    # Needs to predict the distribution over the output, NOT the logits.
    raise NotImplementedError

  def loss_func(self, x, y, mask, data_mask=None):
    raise NotImplementedError

  def load(self, path):
    self.load_state_dict(torch.load(osp.join(path, "best_model.pt"), map_location=self.device))
    self.eval()

  def subsample_mask(self, mask):
    # Subsample by uniformly selecting removal probability, each feature
    # has that probability of being removed.
    # Multiply by true mask since it may be missing values to begin with.
    return (torch.rand_like(mask) > torch.rand_like(mask[:, :1])).float()*mask

  @torch.no_grad()
  def calc_val_dict(self, val_loader, metric_f):
    val_metric = 0
    for x, y, m_data in val_loader:
      x = x.to(self.device)
      y = y.to(self.device)
      m_data = m_data.to(self.device)
      val_metric += metric_f(self.predict(x, m_data), y)/len(val_loader)
    return val_metric, {"Predictive Metric": val_metric}

  def fit(self, train_data, val_data, save_path, metric_f):
    # Main part of training.
    self.fit_parameters(train_data, val_data, save_path, metric_f)
    # End the training, same for all models, finish training by running
    # zero acquisition on the val set and saving the value.
    # We don't use self.load here, since other methods might have
    # special loading requirements, so we just load the parameters and set
    # the model to eval mode.
    self.load_state_dict(torch.load(osp.join(save_path, "best_model.pt"), map_location=self.device))
    self.eval()
    self.post_fitting_tasks(train_data, val_data, save_path, metric_f)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    val_auc = self.run_zero_acquisition(val_loader, metric_f)
    torch.save(val_auc, osp.join(save_path, "val_auc.pt"))
    print(f"\nTraining complete, Zero Acquisition AUC: {val_auc:.3f}")

  def fit_parameters(self, train_data, val_data, save_path, metric_f):
    optimizer = Adam(self.parameters(), lr=self.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=lr_factor,
                                  cooldown=cooldown, min_lr=min_lr, patience=self.patience)
    train_loader = DataLoader(train_data, batch_size=self.batchsize, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

    print("Starting training from scratch")
    for epoch in range(1, self.epochs+1):
      self.train()
      epoch_loss = 0
      for x, y, m_data in train_loader:
        optimizer.zero_grad()
        x = x.to(self.device)
        y = y.to(self.device)
        m_data = m_data.to(self.device)
        loss = self.loss_func(x, y, self.subsample_mask(m_data), m_data)
        loss.backward()
        epoch_loss += loss.item()/len(train_loader)
        optimizer.step()

      self.eval()
      val_metric, val_dict = self.calc_val_dict(val_loader, metric_f)
      scheduler.step(val_metric)
      if val_metric == scheduler.best:
        torch.save(self.state_dict(), osp.join(save_path, "best_model.pt"))

      # Print results of this epoch.
      print(f"\nEpoch: {epoch}/{self.epochs}, Avg Loss: {epoch_loss:.3e}, ", end="")
      print(f"Val Metric: {val_metric:.3f}|{scheduler.best:.3f}", end="")
      for key, value in val_dict.items():
        print(f", {key}: {value:.3f}", end="")

  def post_fitting_tasks(self, train_data, val_data, save_path, metric_f):
    pass

  @torch.no_grad()
  def calculate_acquisition_scores_memory_aware(self, x, mask):
    # We split the validation set if it is too large, since this
    # can be an extremely memory intensive operation.
    if x.shape[0] <= self.batch_limit:
      ids = [torch.arange(x.shape[0])]
    else:
      ids = np.array_split(
        np.arange(x.shape[0]),
        int(x.shape[0] / self.batch_limit),
      )
    scores = []
    for i in ids:
      # Clear cache before and after this, since it is memory intensive.
      self.clear_cache()
      scores.append(self.calculate_acquisition_scores(x[i], mask[i]))
      self.clear_cache()
    scores = torch.cat(scores, dim=0)
    return scores - scores.min()  # Make sure minimum score is 0.

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    # Scores should be positive.
    raise NotImplementedError

  @torch.no_grad()
  def acquire(self, x, mask_acq, mask_data=1.0, return_features=False):
    # We include mask_acq and mask_data, since mask_acq tells us what we have,
    # and mask_data tells us what out dataset lets us acquire.

    # Clear the cache before and after scoring since it can be a memory
    # intensive operation.
    self.eval()
    self.clear_cache()
    feature_scores = self.calculate_acquisition_scores_memory_aware(x, mask_acq*mask_data)
    self.clear_cache()
    feature_scores += 1.0  # Add 1 to all scores to make sure we acquire something.
    feature_scores *= (1.0 - mask_acq)  # Check if feature has been selected already.
    feature_scores *= mask_data  # Based on our dataset, can we even acquire this feature?

    # Anything that has not been acquired that we can will be at least 1.0. But
    # if we acquired everything that can be, we will have all zeros. We still
    # want to believe we have acquired features (even if they can't be), we
    # want to fill the mask. So we finally add a small value to anything not
    # acquired, and then torch argmax selects the first one.
    # NOTE this is for our evaluation only since we want to "acquire" everything
    # at deployment we should be able to measure features.
    feature_scores += (1.0 - mask_acq)*1e-6
    selected = torch.argmax(feature_scores, dim=-1)
    mask_acq = torch.max(mask_acq, F.one_hot(selected, self.num_features).float())
    if return_features:
      return mask_acq, selected
    return mask_acq

  @torch.no_grad()
  def run_zero_acquisition(self, val_loader, metric_f):
    self.eval()
    val_auc = 0
    for x, y, m_data in val_loader:
      x = x.to(self.device)
      y = y.to(self.device)
      m_data = m_data.to(self.device)
      m_acq = torch.zeros_like(m_data)
      val_metrics = [metric_f(self.predict(x, m_acq*m_data), y)]
      for _ in range(self.max_dim):
        m_acq = self.acquire(x, m_acq, m_data)
        val_metrics.append(metric_f(self.predict(x, m_acq*m_data), y))
      val_metrics = np.array(val_metrics)
      val_auc += sklearn_auc(np.arange(self.max_dim+1), val_metrics)/(len(val_loader)*self.max_dim)
    return val_auc
