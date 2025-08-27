"""Implementation of EDDI model. 
Paper: https://arxiv.org/abs/1809.11142
Original code: https://github.com/Microsoft/EDDI
This is a large generative model that estimates the mutual information between
the label and the input features, by taking the expected KL divergence
in the latent space.
"""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from models.base import BaseModel
from models.vae import VAEBase
from models.standard_layers import MLP, StochasticMLP
from models.shared_functions import kl_01_loss, kl_div, nll_con_f, nll_cat_f, nll_mixed_f, sample_cat, sample_mixed


class ContinuousXtoC(nn.Module):
  """The part of EDDI that takes X and mask and gives aggregated C.
  NOTE: This assumes we have continuous features.
  """
  def __init__(self, num_con_features, hidden_dim, c_dim, num_hidden):
    super().__init__()
    self.c_dim = c_dim
    self.num_con_features = num_con_features
    self.E = nn.Parameter(torch.randn(1, num_con_features, c_dim))

    self.s_to_c = MLP(
      in_dim=c_dim+1,
      hidden_dim=hidden_dim,
      out_dim=c_dim,
      num_hidden=num_hidden
    )

  def forward(self, x, mask):
    batchsize = x.shape[0]
    x = x.unsqueeze(-1)
    x = torch.cat([x, x*self.E], dim=-1).view(batchsize*self.num_con_features, -1)
    x = self.s_to_c(x).view(batchsize, self.num_con_features, -1)
    return torch.sum(x*mask.unsqueeze(-1), dim=1)


class CategoricalXtoC(nn.Module):
  """The part of EDDI that takes categorical X and mask and gives aggregated C.
  NOTE: This assumes categorical only features. So we only need an embedding.
  """
  def __init__(self, num_cat_features, c_dim, most_categories):
    super().__init__()
    # We follow the original implementation and embedd everything
    # then take sum multiplied by mask. So we miss the "+1" we have in other
    # models.
    self.E = nn.Embedding(num_cat_features*(most_categories), c_dim)
    self.shift = (torch.arange(num_cat_features)*(most_categories)).unsqueeze(0).long()
    self.shift = nn.Parameter(data=self.shift, requires_grad=False)

  def forward(self, x, mask):
    x = x.long() + self.shift
    x = self.E(x)
    return torch.sum(x*mask.unsqueeze(-1), dim=1)


class EDDI(VAEBase):
  """EDDI base model. Uses generative model to estimate mutual information.
  This is used to score the features and acquire them.
  NOTE: We use one sample from z_dist to train, as is done in EDDI.
  """
  def __init__(self, config):
    super().__init__(config)
    self.latent_dim = config["latent_dim"]
    self.obs_sig = config["sig"]
    self.num_samples_predict = config["num_samples_predict"]
    self.num_samples_acquire = config["num_samples_acquire"]

    if self.input_type == "mixed":
      self.last_con_index = config["num_con_features"]

    # Encoder parts of the model.
    if self.input_type =="mixed" or self.input_type == "continuous":
      self.con_x_to_c = ContinuousXtoC(
        num_con_features=config["num_con_features"],
        hidden_dim=config["hidden_dim_encoder"],
        c_dim=config["c_dim"],
        num_hidden=int(config["num_hidden_encoder"]/2),
      )

    if self.input_type == "mixed" or self.input_type == "categorical":
      self.most_categories = config["most_categories"]
      self.cat_x_to_c = CategoricalXtoC(
        num_cat_features=config["num_cat_features"],
        c_dim=config["c_dim"],
        most_categories=config["most_categories"]
      )

    self.c_to_zdist = StochasticMLP(
      in_dim=config["c_dim"],
      hidden_dim=config["hidden_dim_encoder"],
      out_dim=config["latent_dim"],
      # Share the hidden layers with the s to c part of the encoder.
      num_hidden=config["num_hidden_encoder"] - int(config["num_hidden_encoder"]/2)
    )
    self.E_y = nn.Embedding(config["out_dim"], config["c_dim"])

    # Decoder parts of the model.
    self.z_to_h = nn.Sequential(
      MLP(
        in_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim_decoder"],
        out_dim=config["hidden_dim_decoder"],
        num_hidden=config["num_hidden_decoder"]-1,
      ),
      nn.ReLU(),
      nn.BatchNorm1d(config["hidden_dim_decoder"]),
    )
    self.h_to_y = nn.Linear(config["hidden_dim_decoder"], self.out_dim)
    self.h_to_x = nn.Linear(config["hidden_dim_decoder"], config["num_con_features"]+config["num_cat_features"]*config["most_categories"])

  def x_to_c(self, x, mask):
    if self.input_type == "continuous":
      return self.con_x_to_c(x, mask)
    elif self.input_type == "categorical":
      return self.cat_x_to_c(x, mask)
    elif self.input_type == "mixed":
      x_con = x[:, :self.last_con_index]
      mask_con = mask[:, :self.last_con_index]

      x_cat = x[:, self.last_con_index:]
      mask_cat = mask[:, self.last_con_index:]
      return self.con_x_to_c(x_con, mask_con) + self.cat_x_to_c(x_cat, mask_cat)
    else:
      raise ValueError(f"Unknown input type: {self.input_type}. Must be continuous, categorical or mixed.")

  def predict(self, x, mask):
    z_mu, z_sig = self.c_to_zdist(self.x_to_c(x, mask))
    z_samples = Normal(z_mu, z_sig).rsample([self.num_samples_predict])
    z_samples = z_samples.reshape(-1, z_mu.shape[-1])
    log_preds = F.log_softmax(self.h_to_y(self.z_to_h(z_samples)), dim=-1)
    log_preds = log_preds.view(self.num_samples_predict, -1, self.out_dim)
    log_preds = torch.logsumexp(log_preds, dim=0) - np.log(self.num_samples_predict)
    return torch.exp(log_preds)

  def loss_func(self, x, y, mask, data_mask=None):
    # We follow the EDDI implementation and paper. Where they do not train PVAE
    # for reconstructing features or labels that are masked out. So whilst it would make
    # sense to use the data mask in nll_x, and no mask in nll_y, we follow
    # the original.
    mask_y = torch.bernoulli(torch.full_like(y.float(), 0.5))
    z_mu, z_sig = self.c_to_zdist(self.x_to_c(x, mask) + mask_y.unsqueeze(-1)*self.E_y(y))
    hidden = self.z_to_h(z_mu + z_sig*torch.randn_like(z_mu))
    nll_x = self.nll_f(x, self.h_to_x(hidden), mask)  # Include the subsampled mask here and not data_mask, following original.
    nll_y = torch.mean(F.cross_entropy(self.h_to_y(hidden), y, reduction="none")*mask_y)  # Include the mask_y here, following original.
    return nll_x + nll_y + kl_01_loss(z_mu, z_sig)

  @torch.no_grad()
  def calc_val_dict(self, val_loader, metric_f):
    val_metric = 0
    elbo = 0
    kl = 0
    nll_x = 0
    nll_y = 0
    for x, y, m_data in val_loader:
      x = x.to(self.device)
      y = y.to(self.device)
      m_data = m_data.to(self.device)

      # Predict y based only on x.
      preds = self.predict(x, m_data)
      val_metric += metric_f(preds, y)/len(val_loader)

      # Calculate ELBO.
      z_mu, z_sig = self.c_to_zdist(self.x_to_c(x, m_data) + self.E_y(y))
      hidden = self.z_to_h(z_mu + z_sig*torch.randn_like(z_mu))
      nll_x_tmp = self.nll_f(x, self.h_to_x(hidden), m_data)
      nll_y_tmp = F.cross_entropy(self.h_to_y(hidden), y)
      kl_tmp = kl_01_loss(z_mu, z_sig)
      elbo -= (nll_x_tmp + nll_y_tmp + kl_tmp)/len(val_loader)
      nll_x += nll_x_tmp/(len(val_loader)*self.num_features)
      nll_y += nll_y_tmp/len(val_loader)
      kl += kl_tmp/(len(val_loader)*self.latent_dim)
    return elbo, {"Predictive Metric": val_metric, "KL": kl, "NLL X": nll_x, "NLL Y": nll_y}

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    scores = torch.zeros_like(mask)
    batchsize = x.shape[0]

    x = torch.repeat_interleave(x, self.num_samples_acquire, dim=0)
    mask = torch.repeat_interleave(mask, self.num_samples_acquire, dim=0)
    c_xo = self.x_to_c(x, mask)
    z_mu_xo, z_sig_xo = self.c_to_zdist(c_xo)

    hidden = self.z_to_h(z_mu_xo + z_sig_xo*torch.randn_like(z_mu_xo))
    x_samples = self.sample_from_recon(self.h_to_x(hidden))
    y_samples = sample_cat(self.h_to_y(hidden), self.out_dim).squeeze(-1).long()
    del hidden  # Delete and clear as much as possible.

    for feature in range(mask.shape[-1]):
      self.clear_cache()
      x_io = x.clone()
      mask_tmp = mask.clone()
      mask_tmp[:, feature] = 1.0
      x_io[:, feature] = x_samples[:, feature]

      c_xio = self.x_to_c(x_io, mask_tmp)
      z_mu_xio, z_sig_xio = self.c_to_zdist(c_xio)
      kl1 = kl_div(z_mu_xio, z_sig_xio, z_mu_xo, z_sig_xo).view(batchsize, self.num_samples_acquire)
      kl1 = torch.mean(kl1, dim=-1)

      c_y = self.E_y(y_samples)
      z_mu_oy, z_sig_oy = self.c_to_zdist(c_xo + c_y)
      z_mu_oiy, z_sig_oiy = self.c_to_zdist(c_xio + c_y)
      kl2 = kl_div(z_mu_oiy, z_sig_oiy, z_mu_oy, z_sig_oy).view(batchsize, self.num_samples_acquire)
      kl2 = torch.mean(kl2, dim=-1)

      scores[:, feature] = kl1 - kl2
    return scores

