"""Ablation where we carry out our model in feature space.
We use gradients of features, NOT the latent components. This also means that
we are using a conditional distribution of the features using a VAE.

We load in the VAE, take samples and differentiate the predictor with respect to
the features, we are also limited to use datasets with continuous features only.
It doesn't make mathematical sense to differentiate with respect to categorical 
features, but still is possible.

We are using the VAE and MLP as predictors rather than the SEFA architecture.
"""

import torch
from torch.distributions.normal import Normal

from models.vae import VAE


class FeatureSpaceAblation(VAE):
  def __init__(self, config):
    super().__init__(config)
    self.num_acquisition_samples = 200  # Set to the same as our model.

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    # Take samples from VAE, fill these for each x. Then take the gradients
    # and score accordingly.

    # Sample from the VAE.
    current_preds = self.predict(x, mask)  # Shape = (B, |y|), these are used for prob weighting.

    z_mu, z_sig = self.encoder(x, mask)
    z_samples = Normal(z_mu, z_sig).sample([self.num_acquisition_samples])
    z_samples = z_samples.transpose(0, 1).reshape(-1, self.latent_dim)
    x_samples = self.sample_from_recon(self.decoder(z_samples))

    # Replace the missing values with the sampled values.
    # x and mask are shape (BS, d), can be reshaped to (B, S, d) later.
    x = torch.repeat_interleave(x, self.num_acquisition_samples, dim=0)
    mask = torch.repeat_interleave(mask, self.num_acquisition_samples, dim=0)
    x = mask*x + (1-mask)*x_samples
    mask = torch.ones_like(mask)

    with torch.enable_grad():
      x.requires_grad_(True)
      sampled_preds = self.predict(x, mask)
      preds_sum = torch.sum(sampled_preds, dim=0)
      scores = 0
      for c in range(self.out_dim):
        grads = torch.autograd.grad(preds_sum[c], x, retain_graph=(c!=self.out_dim-1))[0]
        with torch.no_grad():
          grads = grads.view(-1, self.num_acquisition_samples, self.num_features)
          grads = torch.abs(grads)
          grads = grads/(torch.sum(grads, dim=-1, keepdim=True) + 1e-8)
          grads = torch.mean(grads, dim=1)
          scores += grads * current_preds[:, c:c+1]

    return scores