"""Implementation of SEFA."""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from models.base import BaseModel
from models.standard_layers import MLP
from models.constants import log_eps, min_sig
from models.shared_functions import kl_01_loss




class BlockLinear(nn.Module):
  """Individual linear layer in the partitioned model structure.
  in_dims: int, the number of input features per block, must be the same for
                all blocks (2 for SEFA [x_i*m_i, m_i])
  out_dims: int, the number of output features per block, must be the same for
                 all blocks
  num_blocks: int, the number of blocks in the block matrix (number of continuous
                 features for SEFA)
  """
  def __init__(self, in_dims, out_dims, num_blocks):
    super().__init__()
    # Initialize the weights and biases accoring to Kaiming Uniform, as per
    # PyTorch Linear initialisation (this improves learning significantly). See:
    # https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
    # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
    init_bound = 1/(in_dims**0.5)
    self.weight = nn.Parameter(2*init_bound*torch.rand(num_blocks, out_dims, in_dims) - init_bound)
    self.bias = nn.Parameter(2*init_bound*torch.rand(num_blocks*out_dims) - init_bound)
    self.in_dims = in_dims
  
  def forward(self, x):
    # The input must be shape batch, num_groups*in_dims, we reshape to be used
    # with einsum. We then push it through einsum and reshape back.
    # In the einsum shapes, b = number of blocks, o = out dim per block,
    # i = in dim per block, x = batch size.
    batchsize = x.shape[0]
    x = x.view(batchsize, -1, self.in_dims)
    return torch.einsum("boi,xbi->xbo", self.weight, x).view(batchsize, -1) + self.bias


class ContinuousBlockStochasticMLP(nn.Module):
  """Block MLP where we output a mean and diagonal standard deviation.
  This assumes all features are continuous, so all are size 1. We also include
  their mask in the input, so each block is size 2.
  We use softplus to convert a real number to a positive number for the standard
  deviations.
  num_con_features: int, the number of continuous features, i.e. number of blocks
  hidden_dim: int, the number of hidden units PER block
  out_dims: int, the number of output features PER block
  num_hidden: int, the number of hidden layers
  """
  def __init__(self, num_con_features, hidden_dim, out_dims, num_hidden):
    super().__init__()
    # Set up the network. Each feature is concatenated with its mask, so each
    # block is size 2.
    if num_hidden == 0:
      hidden_dim = 2
      self.to_final_hidden = nn.Identity()
    else:
      self.to_final_hidden = [BlockLinear(2, hidden_dim, num_con_features)]
      self.to_final_hidden.append(nn.ReLU())
      self.to_final_hidden.append(nn.BatchNorm1d(hidden_dim*num_con_features))
      for _ in range(num_hidden-1):
        self.to_final_hidden.append(BlockLinear(hidden_dim, hidden_dim, num_con_features))
        self.to_final_hidden.append(nn.ReLU())
        self.to_final_hidden.append(nn.BatchNorm1d(hidden_dim*num_con_features))
      self.to_final_hidden = nn.Sequential(*self.to_final_hidden)

    self.final_hidden_to_mu = nn.Sequential(
      BlockLinear(hidden_dim, out_dims, num_con_features),
      nn.BatchNorm1d(out_dims*num_con_features)
    )
    self.final_hidden_to_sig = nn.Sequential(
      BlockLinear(hidden_dim, out_dims, num_con_features),
      nn.BatchNorm1d(out_dims*num_con_features)
    )

  def forward(self, x, mask):
    batchsize = x.shape[0]
    x = x.unsqueeze(-1)
    mask = mask.unsqueeze(-1)
    x = torch.cat([x*mask, mask], dim=-1).view(batchsize, -1)
    final_hidden = self.to_final_hidden(x)
    mu = self.final_hidden_to_mu(final_hidden)
    sig = F.softplus(self.final_hidden_to_sig(final_hidden)) + min_sig
    return mu, sig


class CategoricalBlockStochasticMLP(nn.Module):
  """Block MLP where we can split the output into a mean and standard deviation.
  Here the inputs are all integers represented as floats. Since all features are 
  independently encoded for categorical, we don't need a block MLP, we just need
  block embeddings.
  A lot of parameters that are never used, because some features do not have
  the maximum number of categories.
  """
  def __init__(self, num_cat_features, out_dims, most_categories):
    super().__init__()
    # We include +1 for a missing mask value.
    # This shift parameters means the embeddings can all be next to each other
    # so feature 1 only has access to first most_categories+1 embeddings,
    # feature 2 to the next most_categories+1 and so on.
    # Means we do not have to for loop over features, and make many embeddings.
    self.mu_embeddings = nn.Embedding(num_cat_features*(most_categories+1), out_dims)
    self.presig_embeddings = nn.Embedding(num_cat_features*(most_categories+1), out_dims)
    self.shift = (torch.arange(num_cat_features)*(most_categories+1)).unsqueeze(0)
    self.shift = nn.Parameter(data=self.shift, requires_grad=False)

  def forward(self, x, mask):
    batchsize = x.shape[0]
    x = ((x+1.0)*mask).long() + self.shift
    mu = self.mu_embeddings(x).view(batchsize, -1)
    presig = self.presig_embeddings(x).view(batchsize, -1)
    return mu, F.softplus(presig) + min_sig


class SEFA(BaseModel):
  """Our model.
  NOTE: We do not use the shared input layers that other models use.
  """
  def __init__(self, config):
    super().__init__(config)
    self.latent_dim = config["latent_dim"]  # This is PER feature needed to initialise encoders.
    self.beta = config["beta"]
    self.num_samples_train = config["num_samples_train"]
    self.num_samples_predict = config["num_samples_predict"]
    self.num_samples_acquire = config["num_samples_acquire"]
    self.last_con_index = config["num_con_features"]
    # The three properties below are changed for ablations only.
    self.normalize_grads = True
    self.prob_weighting = True
    self.deterministic_encoding = False

    self.predictor = MLP(
      in_dim=config["latent_dim"]*self.num_features,
      hidden_dim=config["hidden_dim_predictor"],
      out_dim=self.out_dim,
      num_hidden=config["num_hidden_predictor"]
    )

    if self.input_type == "mixed" or self.input_type == "continuous":
      self.con_encoder = ContinuousBlockStochasticMLP(
        num_con_features=config["num_con_features"],
        hidden_dim=config["hidden_dim_encoder"],
        out_dims=config["latent_dim"],
        num_hidden=config["num_hidden_encoder"],
      )

    if self.input_type == "mixed" or self.input_type == "categorical":
      self.cat_encoder = CategoricalBlockStochasticMLP(
        num_cat_features=config["num_cat_features"],
        out_dims=config["latent_dim"],
        most_categories=config["most_categories"],
      )

  def encode(self, x, mask):
    if self.input_type == "continuous":
      return self.con_encoder(x, mask)
    elif self.input_type == "categorical":
      return self.cat_encoder(x, mask)
    elif self.input_type == "mixed":
      x_con = x[:, :self.last_con_index]
      mask_con = mask[:, :self.last_con_index]
      z_mu_con, z_sig_con = self.con_encoder(x_con, mask_con)

      x_cat = x[:, self.last_con_index:]
      mask_cat = mask[:, self.last_con_index:]
      z_mu_cat, z_sig_cat = self.cat_encoder(x_cat, mask_cat)

      z_mu = torch.cat([z_mu_con, z_mu_cat], dim=-1)
      z_sig = torch.cat([z_sig_con, z_sig_cat], dim=-1)
      return z_mu, z_sig
    else:
      raise ValueError(f"Unknown input type: {self.input_type}. Must be continuous, categorical or mixed.")

  def log_likelihood(self, x, mask, num_samples=100):
    z_mu, z_sig = self.encode(x, mask)
    if self.deterministic_encoding:
      samples = z_mu.unsqueeze(0)
      num_samples = 1
    else:
      samples = Normal(z_mu, z_sig).rsample([num_samples])
    samples = samples.reshape(-1, z_mu.shape[-1])
    log_preds = F.log_softmax(self.predictor(samples), dim=-1)
    log_preds = log_preds.view(num_samples, -1, self.out_dim)
    log_preds = torch.logsumexp(log_preds, dim=0) - np.log(num_samples)
    return log_preds, z_mu, z_sig

  def predict(self, x, mask):
    # Predict has to give the distribution, not logits.
    return torch.exp(self.log_likelihood(x, mask, self.num_samples_predict)[0])

  def loss_func(self, x, y, mask, data_mask=None):
    log_likelihood, z_mu, z_sig = self.log_likelihood(x, mask, self.num_samples_train)
    return F.nll_loss(log_likelihood, y) + self.beta*kl_01_loss(z_mu, z_sig)

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    z_mu, z_sig = self.encode(x, mask)
    if self.deterministic_encoding:
      num_samples = 1
      samples = z_mu.unsqueeze(0)
    else:
      num_samples = self.num_samples_acquire
      samples = Normal(z_mu, z_sig).sample([num_samples])
    with torch.enable_grad():
      samples.requires_grad_(True)
      preds = F.log_softmax(self.predictor(samples.view(-1, z_mu.shape[-1])), dim=-1)
      preds = preds.view(num_samples, -1, self.out_dim)
      preds = torch.logsumexp(preds, dim=0) - np.log(num_samples)
      preds = torch.exp(preds)
      # Calculate scores.
      # NOTE: we have a for loop over the labels here which is slower than
      # creating the full Jacobian with functorch.jacrev or
      # torch.autograd.functional.jacobian for example.
      # However this is currently weighed against the memory requirements
      # for storing a full Jacobian. This can be changed if beneficial.
      preds_sum = torch.sum(preds, dim=0)  # Take sum so we have scalars for torch autograd.
      scores = 0
      for c in range(self.out_dim):
        grads = torch.autograd.grad(preds_sum[c], samples, retain_graph=(c!=self.out_dim-1))[0]
        with torch.no_grad():
          grads = grads.view(num_samples, -1, self.num_features, self.latent_dim)
          grads = torch.sum(grads**2, dim=-1)**0.5
          if self.normalize_grads:
            grads = grads/(torch.sum(grads, dim=-1, keepdim=True) + 1e-8)
          grads = torch.mean(grads, dim=0)
          if self.prob_weighting:
            scores += grads*preds[:, c:c+1]  # NOTE this is if we want to weight by p(Y=c | x_O)
          else:
            scores += grads/self.out_dim  # NOTE this is if we do not do probability weighting.
    return scores

  @torch.no_grad()
  def calc_val_dict(self, val_loader, metric_f):
    val_metric = 0
    ib = 0
    for x, y, mask in val_loader:
      x = x.to(self.device)
      y = y.to(self.device)
      mask = mask.to(self.device)
      log_likelihood, z_mu, z_sig = self.log_likelihood(x, mask, self.num_samples_predict)
      val_metric += metric_f(torch.exp(log_likelihood), y)/len(val_loader)
      ib += kl_01_loss(z_mu, z_sig)/(len(val_loader)*self.latent_dim*self.num_features)
    val_auc = self.run_zero_acquisition(val_loader, metric_f)
    return val_auc, {"Predictive Metric": val_metric, "InfoBottleneck": ib, "Val Auc": val_auc}
