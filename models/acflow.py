"""Implementation of ACFlow, that can be used as part of GSMRL
or also as its own generative model for AFA.
Papers:
https://arxiv.org/abs/1909.06319
https://arxiv.org/abs/2006.07701
Code:
https://github.com/lupalab/ACFlow
https://github.com/lupalab/ACFlow-DFA
"""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from models.constants import log_eps
from models.base import BaseModel
from models.standard_layers import MLP, StochasticMLP


class RealNVPLayer(nn.Module):
  def __init__(self, num_features, num_conditioning_features, hidden_dim, num_hidden):
    """NOTE ACFlow fails on categorical data, since change of variables can't
    apply to non continuous data. We will pretend it is in a continuous space.
    We could also model it using one-hot encodings, treating the space as the
    probability distributions of the classes.
    ___/\___________/\___________/\___________/\___________/\___________/\___
       1            2            3            4            5            6
    Effectively we are treating the distribution as the above. In a continuous
    space, in the limit of Dirac delta functions.
    """
    super().__init__()
    self.za_dim = int(num_features / 2)
    self.zb_dim = num_features - self.za_dim
    # The transforms use zb, and are conditioned on [xo, m_obs, m_data, y]
    #conditioning_features = 3*num_features + num_classes, conditioned on [xo*m_obs, m_obs, m_data, y_onehot]
    in_dim = self.zb_dim + num_conditioning_features
    self.scale_network = nn.Sequential(
      MLP(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=self.za_dim,
        num_hidden=num_hidden,
      ),
      nn.Tanh(),  # Use a tanh at the end to keep everything stable.
    )
    self.shift_network = nn.Sequential(
      MLP(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=self.za_dim,
        num_hidden=num_hidden,
      ),
      nn.Tanh(),
    )
    self.perm = np.random.choice(num_features, num_features, replace=False)
    self.inv_perm = np.argsort(self.perm)

  def split(self, z):
    # Take the vector and split it according to the permutations.
    z = z[:, self.perm]
    return z[:, :self.za_dim], z[:, self.za_dim:]

  def recombine(self, za, zb):
    # Concatenate, and inverse the permutation.
    return torch.cat([za, zb], dim=-1)[:, self.inv_perm]

  def forward(self, z, conditioning):
    za, zb = self.split(z)
    input = torch.cat([zb, conditioning], dim=-1)
    s = self.scale_network(input)
    t = self.shift_network(input)
    za = za * torch.exp(s) + t
    z = self.recombine(za, zb)
    log_det = torch.sum(s, dim=-1)
    return z, log_det

  def inverse(self, z, conditioning):
    za, zb = self.split(z)
    input = torch.cat([zb, conditioning], dim=-1)
    s = self.scale_network(input)
    t = self.shift_network(input)
    za = (za - t) * torch.exp(-s)
    z = self.recombine(za, zb)
    log_det = torch.sum(-s, dim=-1)
    return z, log_det


class FlowNetwork(nn.Module):
  def __init__(self, num_features, num_conditioning_features, hidden_dim, num_hidden, num_flow_modules):
    super().__init__()
    self.nvp_layers = nn.ModuleList([
      RealNVPLayer(num_features, num_conditioning_features, hidden_dim, num_hidden) for _ in range(num_flow_modules)
    ])

  def forward(self, z, conditioning):
    log_det = 0.0
    for layer in self.nvp_layers:
      z, log_det_layer = layer.forward(z, conditioning)
      log_det += log_det_layer
    return z, log_det

  def inverse(self, z, conditioning):
    log_det = 0.0
    for layer in reversed(self.nvp_layers):
      z, log_det_layer = layer.inverse(z, conditioning)
      log_det += log_det_layer
    return z, log_det


class PriorNetwork(nn.Module):
  def __init__(self, num_features, num_conditioning_features, hidden_dim, num_hidden):
    super().__init__()
    self.prior_network = StochasticMLP(
      in_dim=num_conditioning_features,
      hidden_dim=hidden_dim,
      out_dim=num_features,
      num_hidden=num_hidden,
    )

  def get_prior(self, conditioning):
    mean, sig = self.prior_network(conditioning)
    return Normal(loc=mean, scale=sig)

  def log_likelihood(self, z, conditioning):
    prior = self.get_prior(conditioning)
    return torch.sum(prior.log_prob(z), dim=-1)

  def conditional_latent_sample(self, conditioning, num_samples):
    # Output is batchsize*num_samples, d. Where we repeat the samples for each
    # batch. i.e. for 3 samples:
    # [x1_sample1, x1_sample2, x1_sample3, x2_sample1, x2_sample2, ...]
    batchsize = conditioning.shape[0]
    prior = self.get_prior(conditioning)
    samples = prior.sample([num_samples])  # Shape is num_samples, batchsize, d
    assert samples.shape[0] == num_samples
    assert samples.shape[1] == batchsize
    samples = torch.transpose(samples, 0, 1)
    assert samples.shape[0] == batchsize
    assert samples.shape[1] == num_samples
    samples = samples.reshape(-1, samples.shape[-1])
    return samples



class ACFlow(BaseModel):

  def __init__(self, config):
    super().__init__(config)
    # Override the basemodel initialization, deleting what we don't need.
    del self.in_dim
    del self.input_layer
    del self.input_type

    # Specific hyperparameters to ACFlow.
    self.lambda_nll = config["lambda_nll"]
    self.lambda_xent_sub = config["lambda_xent_sub"]
    self.lambda_xent_full = config["lambda_xent_full"]
    self.num_samples_acquire = config["num_samples_acquire"]

    # Get the numpy state, so we can set it back after the model is created.
    # We have to use the same numpy seed to ensure constructing the model is
    # determined by the config.
    # Since we use random permutations to construct flow modules.
    current_numpy_state = np.random.get_state()
    np.random.seed(2487)

    # Learnable modules.
    num_conditioning_features = 3*self.num_features+self.out_dim

    self.flow = FlowNetwork(
      num_features=self.num_features,
      num_conditioning_features=num_conditioning_features,
      hidden_dim=config["hidden_dim_flow"],
      num_hidden=config["num_hidden_flow"],
      num_flow_modules=config["num_flow_modules"],
    )

    self.prior = PriorNetwork(
      num_features=self.num_features,
      num_conditioning_features=num_conditioning_features,
      hidden_dim=config["hidden_dim_prior"],
      num_hidden=config["num_hidden_prior"],
    )

    self.num_classes = config["out_dim"]
    self.log_class_probs = config["log_class_probs"].unsqueeze(0).float()
    self.log_class_probs = nn.Parameter(data=self.log_class_probs, requires_grad=False)

    # Set random state back.
    np.random.set_state(current_numpy_state)

  def loss_func(self, x, y, subsampled_mask, data_mask):
    log_pu = self.calc_logits(x, subsampled_mask, data_mask)  # log p(xu | xo, y)
    log_po = self.calc_logits(x, 0.0*subsampled_mask, subsampled_mask)  # log p(xo | y)
    log_pou = log_po + log_pu  # log p(xu | xo, y) + log p(xo | y) = log p(xu, xo | y)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Classification using all observations.
    # log p(xu, xo |y) + log p(y) = log p(xu, xo, y) = log p(y | xo, xu) + log p(xo, xu)
    # could be done with log_pou = self.calc_logits(x, 0.0*subsampled_mask, data_mask)
    # We follow the implementation.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logits = log_pou + self.log_class_probs
    full_cross_entropy = F.cross_entropy(logits, y)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Classification using subsampled observations.
    # log p(xo | y) + log p(y) = log p(y, xo) = log p(y | xo) + log p(xo)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sublogits = log_po + self.log_class_probs
    sub_cross_entropy = F.cross_entropy(sublogits, y)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Likelihood.
    # log p(xu | xo) = log p(xu, xo) - log p(xo)
    # = logsumexp(log p(xu, xo, y)) - logsumexp(log p(xo , y))
    # We can ignore the log p(y) term since it cancels out.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    neg_log_likelihood = -(
      torch.logsumexp(logits, dim=-1) - torch.logsumexp(sublogits, dim=-1)
    ).mean() / self.num_features

    # Full loss with hyperparameter weightings.
    return (self.lambda_nll*neg_log_likelihood 
            + self.lambda_xent_full*full_cross_entropy
            + self.lambda_xent_sub*sub_cross_entropy)

  def predict(self, x, mask):
    # NOTE, here the mask says what we have access to, rather than genuine data.
    # The assumption is that this mask has been properly constructed.
    log_px = self.calc_logits(x, 0.0*mask, mask)  # log p(x | y)
    logits = log_px + self.log_class_probs
    assert logits.ndim == 2, "Logits should be of shape (batchsize, num_classes)"
    assert logits.shape[0] == x.shape[0], "Logits should have same batch size as input"
    assert logits.shape[1] == self.num_classes, "Logits should have num_classes as second dimension"
    return F.softmax(logits, dim=-1)

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    # NOTE, here the mask says what we have access to.
    scores = torch.empty_like(mask)
    batchsize = x.shape[0]
    y_preds = self.predict(x, mask)
    H_y_xo = -torch.sum(y_preds * torch.log(y_preds + log_eps), dim=-1)

    # Generate some samples for the entropy calculation.
    x_samples = self.unconditional_samples(x, mask, mask, self.num_samples_acquire)
    mask = torch.repeat_interleave(mask, self.num_samples_acquire, dim=0)
    # It is possible to avoid this for loop by extending the dimensions
    # even more, so we can run each in parallel, but is very memory intensive.
    for feature in range(self.num_features):
      mask_tmp = mask.clone()
      mask_tmp[:, feature] = 1.0
      new_y_preds = self.predict(x_samples, mask_tmp)
      H_y_xio = -torch.sum(new_y_preds * torch.log(new_y_preds + log_eps), dim=-1)
      H_y_xio = torch.mean(H_y_xio.view(batchsize, self.num_samples_acquire), dim=-1)
      scores[:, feature] = H_y_xo - H_y_xio
    return scores

  def get_xu_conditioning(self, x, y, m_obs, m_data, forward):
    m_obs = m_obs * m_data  # Make sure we know what observations are genuine.
    xu = x * (1 - m_obs) * m_data
    xo = x * m_obs
    y = F.one_hot(y.long(), num_classes=self.num_classes).float()
    # If forward we have to consider what the data says is present.
    # If backwards we are generating samples, so in this case we have to
    # set mask to 1, so it assumes everything can be generated. Crucially
    # we still have m_obs = m_obs * m_data before, so the model knows which
    # observations are genuine.
    m_data = m_data if forward else torch.ones_like(m_data)
    conditioning = torch.cat([xo, m_obs, m_data, y], dim=-1)
    return xu, conditioning

  def conditional_forward(self, x, y, m_obs, m_data):
    m_obs = m_obs * m_data
    xu, conditioning = self.get_xu_conditioning(x, y, m_obs, m_data, forward=True)
    z, log_det = self.flow.forward(xu, conditioning)
    prior_log_likelihood = self.prior.log_likelihood(z, conditioning)
    return prior_log_likelihood + log_det

  def conditional_samples(self, x, y, m_obs, m_data, num_samples):
    # Sampling conditioned on x_o, m_o and y.
    # NOTE: Returns shape batch*num_samples, d as:
    # [x1_sample1, x1_sample2, x1_sample3, x2_sample1, x2_sample2, ...]
    m_obs = m_obs * m_data
    _, conditioning = self.get_xu_conditioning(x, y, m_obs, m_data, forward=False)
    z_u = self.prior.conditional_latent_sample(conditioning, num_samples)
    conditioning = torch.repeat_interleave(conditioning, num_samples, dim=0)
    x_u, _ = self.flow.inverse(z_u, conditioning)
    # Fill in where we did not have "genuine" observations.
    x = torch.repeat_interleave(x, num_samples, dim=0)
    m_obs = torch.repeat_interleave(m_obs, num_samples, dim=0)
    x_sampled =  + m_obs*x + (1 - m_obs)*x_u
    return x_sampled

  def unconditional_samples(self, x, m_obs, m_data, num_samples):
    # Sampling without y condition. So we sample from p(xu, y | xo) and
    # disregard samples.
    # p(xu, y | xo) = p(xu | xo, y) * p(y | xo)
    # So we can sample y from p(y | xo) and then sample xu from p(xu | xo, y).
    # NOTE: Returns shape batch*num_samples, d as:
    # [x1_sample1, x1_sample2, x1_sample3, x2_sample1, x2_sample2, ...]
    m_obs = m_obs * m_data
    py = self.predict(x, m_obs)
    y_samples = torch.multinomial(py, num_samples, replacement=True).view(-1)
    x = torch.repeat_interleave(x, num_samples, dim=0)
    m_obs = torch.repeat_interleave(m_obs, num_samples, dim=0)
    m_data = torch.repeat_interleave(m_data, num_samples, dim=0)
    samples = self.conditional_samples(x, y_samples, m_obs, m_data, num_samples=1)
    return samples

  def calc_logits(self, x, m_obs, m_data):
    # Calculates log p(xu | xo, y) for each possible y.
    # It subs in all possible classes, and then reshapes the output.
    # So that we get log p(xu | xo, y), for each possible class. We can
    # then use Bayes' to get log p(y | xo, xu).
    # Shape out = batch, num_classes.
    batchsize = x.shape[0]
    x = torch.repeat_interleave(x, self.num_classes, dim=0)
    m_obs = torch.repeat_interleave(m_obs, self.num_classes, dim=0)
    m_data = torch.repeat_interleave(m_data, self.num_classes, dim=0)
    fake_y = torch.arange(self.num_classes, device=self.device).repeat(batchsize)
    logits = self.conditional_forward(x, fake_y, m_obs, m_data)
    assert logits.shape[0] == batchsize * self.num_classes
    assert logits.ndim == 1
    return logits.reshape(batchsize, self.num_classes)