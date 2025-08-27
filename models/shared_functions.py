"""Utils functions for the models, like KL divergence, NLL, etc.
Functions we do not expect in an acquisition model by default, but are shared.
"""

import numpy as np

import torch
import torch.nn.functional as F

from models.constants import log_eps, half_log_2pi



##############################
##############################
####### KL Divergences #######
##############################
##############################

def kl_01_loss(z_mu, z_sig):
  # KL divergence between a normal distribution and a standard normal.
  # KL(N(mu, sig) || N(0, 1))
  # Include mean at the end since this is a loss function.
  kl = z_sig**2 + z_mu**2 -2*torch.log(z_sig + log_eps) - 1
  return torch.mean(torch.sum(0.5*kl, dim=-1))


def kl_div(mu0, sig0, mu1, sig1):
  # KL divergence between two normal distributions.
  # KL(N0 || N1)
  # Do not include mean at the end since we need to access the batch KL values.
  kl = 2*torch.log(sig1 + log_eps) - 2*torch.log(sig0 + log_eps)
  kl += (sig0**2 + (mu0 - mu1)**2)/(sig1**2) - 1
  return torch.sum(0.5*kl, dim=-1)


##############################
##############################
######### VAE Utils ##########
##############################
##############################

def nll_con_f(x, recon_mu, mask, sig):
  # Reconstruction nll for continuous data.
  # Sig is not a parameter, so we can use numpy.
  out = 0.5*((x - recon_mu)/sig)**2 + np.log(sig) + half_log_2pi
  return torch.mean(torch.sum(out*mask, dim=-1))


def nll_cat_f(x, recon_logits, mask):
  # Reconstruction nll for categorical data. Receive logits of shape
  # (batchsize, num_cat_features*most_categories). So we reshape
  # to (batchsize*num_cat_features, most_categories) and apply cross entropy.
  # Then reshape to (batchsize, num_cat_features), sum and average.
  batchsize = x.shape[0]
  num_features = mask.shape[-1]
  recon_logits = recon_logits.reshape(batchsize*num_features, -1)
  x = x.reshape(-1).long()
  out = F.cross_entropy(recon_logits, x, reduction="none").view(batchsize, -1)
  return torch.mean(torch.sum(out*mask, dim=-1))


def nll_mixed_f(x, recon, mask, sig, last_con_index):
  # Reconstruction nll for mixed types.
  x_con = x[:, :last_con_index]
  recon_con = recon[:, :last_con_index]
  mask_con = mask[:, :last_con_index]
  nll_con = nll_con_f(x_con, recon_con, mask_con, sig)

  x_cat = x[:, last_con_index:]
  recon_cat = recon[:, last_con_index:]
  mask_cat = mask[:, last_con_index:]
  nll_cat = nll_cat_f(x_cat, recon_cat, mask_cat)
  return nll_con + nll_cat


def sample_cat(recon_cat, most_categories):
  # Given reconstruction logits shape: batchsize, num_cat_features*most_categories
  # we convert to categorical and sample.
  batchsize = recon_cat.shape[0]
  recon_cat = F.softmax(recon_cat.reshape(-1, most_categories), dim=-1)
  return torch.multinomial(recon_cat, 1).view(batchsize, -1).float()


def sample_mixed(recon, last_con_index, most_categories):
  recon_con = recon[:, :last_con_index]  # Continuous part is the mean.
  recon_cat = recon[:, last_con_index:]
  recon_cat = sample_cat(recon_cat, most_categories)
  return torch.cat([recon_con, recon_cat], dim=-1)