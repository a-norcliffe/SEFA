"""Simple VAE model, taking one step back from EDDI.
The model has a predictive part and a generative part. These are trained
together. To acquire we use the expression for the mutual information:
R(x_o, i) = E_{p(x_i|x_o)}[ D_{KL}(p(y|x_i, x_o) || p(y|x_o)) ]
"""

import torch
import torch.nn.functional as F

from torch.distributions import Normal

from models.constants import log_eps
from models.base import BaseModel
from models.standard_layers import MLP, StochasticMLP
from models.shared_functions import kl_01_loss, nll_con_f, nll_cat_f, nll_mixed_f, sample_cat, sample_mixed



class VAEBase(BaseModel):
  """The VAE base only contains functions for calculating reconstruction loss and sampling.
  When we have mixed inputs."""
  def __init__(self, config):
    super().__init__(config)

  def nll_f(self, x, recon, mask):
    if self.input_type == "continuous":
      return nll_con_f(x, recon, mask, self.obs_sig)
    elif self.input_type == "categorical":
      return nll_cat_f(x, recon, mask)
    elif self.input_type == "mixed":
      return nll_mixed_f(x, recon, mask, self.obs_sig, self.last_con_index)
    else:
      raise ValueError(f"Unknown input type: {self.input_type}. Must be continuous, categorical or mixed.")

  def sample_from_recon(self, recon):
    if self.input_type == "continuous":
      return recon  # For continuous we just give the mu values.
    elif self.input_type == "categorical":
      return sample_cat(recon, self.most_categories)
    elif self.input_type == "mixed":
      return sample_mixed(recon, self.last_con_index, self.most_categories)
    else:
      raise ValueError(f"Unknown input type: {self.input_type}. Must be continuous, categorical or mixed.")


class VAE(VAEBase):
  """Vanilla VAE baseline. Has a predictive model and a VAE generative model
  for x. The predictive model is trained to predict y from x. The VAE model is
  trained with ELBO. Acquisition is done with the mutual information expression.
  Decoder predicts mu for continuous and logits for categorical.
  """
  def __init__(self, config):
    super().__init__(config)
    self.latent_dim = config["latent_dim"]
    self.obs_sig = config["sig"]
    self.num_samples_acquire = config["num_samples_acquire"]

    if self.input_type == "mixed":
      self.last_con_index = config["num_con_features"]
    if self.input_type == "categorical" or self.input_type == "mixed":
      self.most_categories = config["most_categories"]

    self.predictor_layers = MLP(
      in_dim=self.in_dim,
      hidden_dim=config["hidden_dim_predictor"],
      out_dim=self.out_dim,
      num_hidden=config["num_hidden_predictor"],
    )
    self.encoder_layers = StochasticMLP(
      in_dim=self.in_dim,
      hidden_dim=config["hidden_dim_encoder"],
      out_dim=config["latent_dim"],
      num_hidden=config["num_hidden_encoder"],
    )
    self.decoder = MLP(
      in_dim=config["latent_dim"],
      hidden_dim=config["hidden_dim_decoder"],
      out_dim=config["num_con_features"]+config["num_cat_features"]*config["most_categories"],
      num_hidden=config["num_hidden_decoder"],
    )

  def predictor(self, x, mask):
    return self.predictor_layers(self.input_layer(x, mask))

  def predict(self, x, mask):
    return F.softmax(self.predictor(x, mask), dim=-1)

  def encoder(self, x, mask):
    return self.encoder_layers(self.input_layer(x, mask))

  def loss_func(self, x, y, mask, data_mask):
    pred_loss = F.cross_entropy(self.predictor(x, mask), y)
    z_mu, z_sig = self.encoder(x, mask)
    recon = self.decoder(z_mu + z_sig*torch.randn_like(z_mu))
    nll = self.nll_f(x, recon, mask)
    kl_loss = kl_01_loss(z_mu, z_sig)
    # Don't need to weight the losses, since they are for separate paramters.
    return (nll + kl_loss) + pred_loss

  @torch.no_grad()
  def calc_val_dict(self, val_loader, metric_f):
    val_metric = 0
    elbo = 0
    kl = 0
    nll_x = 0
    for x, y, m_data in val_loader:
      x = x.to(self.device)
      y = y.to(self.device)
      m_data = m_data.to(self.device)

      # Predict y based only on x.
      val_metric += metric_f(self.predict(x, m_data), y)/len(val_loader)

      # Calculate ELBO.
      z_mu, z_sig = self.encoder(x, m_data)
      recon = self.decoder(z_mu + z_sig*torch.randn_like(z_mu))
      nll_x_tmp = self.nll_f(x, recon, m_data)
      kl_tmp = kl_01_loss(z_mu, z_sig)
      elbo -= (nll_x_tmp + kl_tmp)/len(val_loader)
      nll_x += nll_x_tmp/(len(val_loader)*self.num_features)
      kl += kl_tmp/(len(val_loader)*self.latent_dim)
    return elbo, {"Predictive Metric": val_metric, "KL": kl, "NLL X": nll_x, "ELBO": elbo}

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    # This calculates mutual information. It uses a for loop over features,
    # which can be slow, but is memory efficient.
    scores = torch.empty_like(mask)
    batchsize = x.shape[0]

    y_preds = self.predict(x, mask)
    H_y_xo = -torch.sum(y_preds * torch.log(y_preds + log_eps), dim=-1)

    z_mu, z_sig = self.encoder(x, mask)
    z_samples = Normal(z_mu, z_sig).sample([self.num_samples_acquire])
    z_samples = z_samples.transpose(0, 1).reshape(-1, self.latent_dim)
    x_samples = self.sample_from_recon(self.decoder(z_samples))

    x = torch.repeat_interleave(x, self.num_samples_acquire, dim=0)
    mask = torch.repeat_interleave(mask, self.num_samples_acquire, dim=0)

    # This is the slow for loop, we could do more interleaving if the memory allows.
    for feature in range(self.num_features):
      x_tmp = x.clone()
      mask_tmp = mask.clone()
      mask_tmp[:, feature] = 1.0
      x_tmp[:, feature] = x_samples[:, feature]

      new_y_preds = self.predict(x_tmp, mask_tmp)
      H_y_xio = -torch.sum(new_y_preds * torch.log(new_y_preds + log_eps), dim=-1)
      H_y_xio = torch.mean(H_y_xio.view(batchsize, self.num_samples_acquire), dim=-1)

      scores[:, feature] = H_y_xo - H_y_xio
    return scores
