"""Implementation of GSMRL model,
Paper: https://arxiv.org/abs/2010.02433
Github: https://github.com/lupalab/GSMRL
Original implementation is in tensorflow, we will implement it in pytorch.

Uses ACFlow which has also been implemented as another AFA model in pytorch.

We have used the below resource for aid in implementing PPO:
https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146
https://medium.com/@z4xia/coding-ppo-from-scratch-with-pytorch-part-4-4-4e21f4a63e5c
https://github.com/ericyangyu/PPO-for-Beginners

We use a PPO base class:
https://arxiv.org/abs/1707.06347
https://en.wikipedia.org/wiki/Proximal_policy_optimization
https://spinningup.openai.com/en/latest/algorithms/ppo.html

And then implement GSMRL as a subclass of PPO. That uses intermediate rewards
and auxiliary information via a surrogate model.
"""

import os.path as osp
import itertools

import numpy as np
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.base import BaseModel
from models.acflow import ACFlow
from models.fixed_mlp import FixedMLP
from models.constants import lr_factor, cooldown, min_lr, log_eps
from models.standard_layers import MLP
from models.rl_utils import IterableIds


class PPO(BaseModel):
  """Simple PPO baseline implementation."""
  def __init__(self, config):
    super().__init__(config)
    self.gamma = config["gamma"]
    self.lambda_gae = config["lambda_gae"]
    self.lambda_entropy = config["lambda_entropy"]
    self.grad_norm = config["grad_norm"]
    self.num_episodes = config["num_episodes"]
    self.num_epochs_per_episode = config["num_epochs_per_episode"]
    self.rollout_batchsize = config["rollout_batchsize"]
    self.optimization_batchsize = config["optimization_batchsize"]
    self.in_dim = self.in_dim + self.num_auxiliary_info

    # The models.
    self.actor_layers = MLP(
      in_dim=self.in_dim,
      hidden_dim=config["hidden_dim"],
      out_dim=self.num_features,
      num_hidden=config["num_hidden_actor"],
    )
    self.critic_layers = MLP(
      in_dim=self.in_dim,
      hidden_dim=config["hidden_dim"],
      out_dim=1,
      num_hidden=config["num_hidden_critic"],
    )

    # Set up a predictive model and a surrogate model if we need one.
    self.get_predictive_model(config)
    self.get_surrogate_model(config)

    # Stuff for printing purposes only.
    self.print_every = 20
    self.eval_every = 5
    self.ema_alpha_fast = 0.1
    self.ema_alpha_slow = 0.005
    self.large_number = 1e6  # For applying masking to the logits.

  def get_surrogate_model(self, config):
    pass

  def get_predictive_model(self, config):
    # Load in fixed mlp and freeze the weights.
    predictive_model_path = osp.join("experiments", "trained_models", config["dataset"], "fixed_mlp")
    predictive_model_config = torch.load(osp.join(predictive_model_path, f"config.pt"))
    self.predictive_model = FixedMLP(predictive_model_config)
    repeat = config["repeat"] if "repeat" in config else 1
    self.predictive_model.load(osp.join(predictive_model_path, f"repeat_{repeat}"))
    self.predictive_model.eval()
    for param in self.predictive_model.parameters():
      param.requires_grad = False

  @torch.no_grad()
  def generate_auxiliary_info(self, x, mask):
    # Baseline does not use any auxiliary_info.
    return None

  @property
  def num_auxiliary_info(self):
    return 0

  @torch.no_grad()
  def prepare_input(self, x, mask, auxiliary_info):
    return self.input_layer(x, mask)

  @torch.no_grad()
  def calculate_reward(self, x, y, auxiliary_info, m_curr, m_next):
    # The baseline PPO is the cross entropy of the pretrained predictor at every
    # step. Assumes the masks have been multiplied by the missing data mask already.
    self.predictive_model.eval()
    logits_preds = self.predictive_model(x, m_next)
    return -F.cross_entropy(logits_preds, y, reduction="none")

  def predict(self, x, mask):
    # We change here from using the surrogate model to just using the built in
    # predictor. This is different from the original implementation, since
    # we have observed that ACFlow give low quality predictions p(y | x_o)
    # (like other generative approaches without dedicated prediction models).
    self.predictive_model.eval()
    return self.predictive_model.predict(x, mask)

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    auxiliary_info = self.generate_auxiliary_info(x, mask)
    input = self.prepare_input(x, mask, auxiliary_info)
    return F.softmax(self.actor_layers(input), dim=-1)

  def get_masked_logpa(self, input, m_curr, m_data):
    logits = self.actor_layers(input)
    # 1 if m_curr = 0.0 (not done before) AND if m_data = 1.0 (available).
    m = (1-m_curr)*m_data
    logits = m*logits + (1-m)*(-self.large_number)
    # Add large number to logits that are not selected before.
    # Therefore, we select stuff we have not selected before uniformly, but only if
    # we have selected everything that was available first. Then there are
    # no gradients associated with these actions, which are done so we can
    # batch train.
    logits = logits + (1-m_curr)*self.large_number
    return F.log_softmax(logits, dim=-1)

  @torch.no_grad()
  def generate_rollouts(self, x_ep, y_ep, m_data_ep):
    # Set up the buffers for the episode. Normally we would have a class
    # for this, but this way we have everything in one place.
    input_buffer = []
    m_data_buffer = []
    m0_buffer = []
    m1_buffer = []
    reward_history = []  # Make it clear we do not return this.
    old_logpa_buffer = []
    v_pred_history = []  # Distinguish from v_target, by calling this history.

    m_curr = torch.zeros_like(m_data_ep)
    for _ in range(self.max_dim):
      auxiliary_info = self.generate_auxiliary_info(x_ep, m_data_ep*m_curr)
      input = self.prepare_input(x_ep, m_data_ep*m_curr, auxiliary_info)

      # Sample an action, but only from available non-selected actions.
      logpa = self.get_masked_logpa(input, m_curr, m_data_ep)
      selected = torch.multinomial(torch.exp(logpa), num_samples=1).squeeze(1)
      selected = F.one_hot(selected, num_classes=self.num_features).float()
      m_next = torch.max(m_curr, selected)

      # Calculate reward.
      reward = self.calculate_reward(
        x_ep, y_ep, auxiliary_info, m_data_ep*m_curr, m_data_ep*m_next,
      )

      # Push everything into the buffers.
      input_buffer.append(input)
      m_data_buffer.append(m_data_ep)
      m0_buffer.append(m_curr)
      m1_buffer.append(m_next)
      reward_history.append(reward)
      old_logpa_buffer.append(torch.sum(logpa*selected, dim=-1))
      v_pred_history.append(self.critic_layers(input).squeeze(-1))

      # Get ready for next acquisition.
      m_curr = m_next

    # After the generated rollouts, we calculate estimates of additional
    # quantities. Such as the advantages.
    input_buffer = torch.cat(input_buffer, dim=0)  # [TB, in_dim+d_auxiliary]
    m_data_buffer = torch.cat(m_data_buffer, dim=0)  # [TB, d]
    m0_buffer = torch.cat(m0_buffer, dim=0)  # [TB, d]
    m1_buffer = torch.cat(m1_buffer, dim=0)  # [TB, d]
    old_logpa_buffer = torch.cat(old_logpa_buffer, dim=0)  # [TB]

    # We use Generalized Advantage Estimation (GAE) to estimate the
    # advantages and value targets. (https://arxiv.org/abs/1506.02438)
    advantage_so_far = 0.0
    advantages_buffer = []
    for t in reversed(range(self.max_dim)):
      if t == self.max_dim-1:
        delta = reward_history[t] - v_pred_history[t]
      else:
        delta = reward_history[t] + self.gamma*v_pred_history[t+1] - v_pred_history[t]
      advantage_so_far = delta + self.gamma*self.lambda_gae*advantage_so_far
      advantages_buffer.insert(0, advantage_so_far)  # Insert at front.
    advantages_buffer = torch.cat(advantages_buffer, dim=0)  # [TB]
    v_target_buffer = advantages_buffer + torch.cat(v_pred_history, dim=0)  # [TB]
    return {
      "input": input_buffer,
      "m_data": m_data_buffer,
      "m0": m0_buffer,
      "m1": m1_buffer,
      "old_logpa": old_logpa_buffer,
      "v_target": v_target_buffer,
      "advantages": advantages_buffer,
      "length": input_buffer.shape[0],
    }

  @torch.enable_grad()
  def train_networks_on_rollouts(self, buffer):
    for _ in range(self.num_epochs_per_episode):
      episode_train_ids = np.random.permutation(buffer["length"])
      episode_train_ids = np.array_split(episode_train_ids, ceil(buffer["length"]/self.optimization_batchsize))
      for ids in episode_train_ids:
        input = buffer["input"][ids]
        m_data = buffer["m_data"][ids]
        m0 = buffer["m0"][ids]
        m1 = buffer["m1"][ids]
        old_logpa = buffer["old_logpa"][ids]
        v_target = buffer["v_target"][ids]
        advantage = buffer["advantages"][ids]

        # Do some training based on stuff from the buffer.

        # The actor part.
        self.optimizer.zero_grad()
        logpa = self.get_masked_logpa(input, m0, m_data)
        pa_entropy = -torch.sum(torch.exp(logpa)*logpa, dim=-1)
        logpa = torch.sum(logpa*(m1 - m0), dim=-1)
        ratio = torch.exp(logpa - old_logpa)
        ratio_clipped = torch.clamp(ratio, min=0.8, max=1.2)
        loss_a = -torch.minimum(advantage*ratio, advantage*ratio_clipped)
        loss_a = torch.mean(loss_a) - self.lambda_entropy*torch.mean(pa_entropy)
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor_layers.parameters(), self.grad_norm)
        self.optimizer.step()

        # The critic part.
        self.optimizer.zero_grad()
        critic_preds = self.critic_layers(input).squeeze(-1)
        loss_c = F.mse_loss(critic_preds, v_target)
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic_layers.parameters(), self.grad_norm)
        self.optimizer.step()

        # Calculate moving averages of the losses.
        if hasattr(self, "loss_actor_avg"):
          self.loss_actor_avg = (1-self.ema_alpha_slow)*self.loss_actor_avg + self.ema_alpha_slow*loss_a.item()
          self.loss_critic_avg = (1-self.ema_alpha_slow)*self.loss_critic_avg + self.ema_alpha_slow*loss_c.item()
        else:
          self.loss_actor_avg = loss_a.item()
          self.loss_critic_avg = loss_c.item()

  def fit_parameters(self, train_data, val_data, save_path, metric_f):
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    self.optimizer = Adam(
      params=itertools.chain(  # Explicitly don't include surrogate model. Or predictive model.
        self.actor_layers.parameters(),
        self.critic_layers.parameters(),
      ),
      lr=self.lr,
    )
    self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=lr_factor,
                                  cooldown=cooldown, min_lr=min_lr, patience=self.patience)
    train_ids_iter = IterableIds(len(train_data), self.rollout_batchsize)

    # Start the training.
    print("Starting training from scratch\n")
    for episode in range(1, self.num_episodes+1):
      self.train()

      # Generate rollouts with subset of data, WITHOUT gradients.
      random_ids = train_ids_iter.next()
      x_ep, y_ep, m_data_ep = train_data[random_ids]
      x_ep = x_ep.to(self.device)
      y_ep = y_ep.to(self.device)
      m_data_ep = m_data_ep.to(self.device)
      buffer = self.generate_rollouts(x_ep, y_ep, m_data_ep)
      # Make sure we have no gradients from the rollouts.
      for v in buffer.values():
        if isinstance(v, int):
          pass
        else:
          assert v.requires_grad is False, "Gradients from PPO rollouts."

      # After generating the rollouts, we optimize on the buffer data
      # for a few epochs.
      self.train_networks_on_rollouts(buffer)

      # Evaluate validation info and update exponential moving averages.
      if episode%self.eval_every == 0:
        self.eval()
        val_metric, _ = self.calc_val_dict(val_loader, metric_f)
        val_auc = self.run_zero_acquisition(val_loader, metric_f)
        # If we have calculated the first metrics, do EMA.
        if hasattr(self, "val_auc_avg"):
          self.val_auc_avg = (1-self.ema_alpha_fast)*self.val_auc_avg + self.ema_alpha_fast*val_auc
        else:
          self.val_auc_avg = val_auc

        self.scheduler.step(val_auc)
        if val_auc == self.scheduler.best:
          torch.save(self.state_dict(), osp.join(save_path, "best_model.pt"))

      if episode%self.print_every == 0:
        print(f"Episode {episode}/{self.num_episodes}, ", end="")
        print(f"EMA Val AUC: {self.val_auc_avg:.3f}|{self.scheduler.best:.3f}, ", end="")
        print(f"EMA Val Metric: {val_metric:.3f}, ", end="")
        print(f"EMA Actor Loss: {self.loss_actor_avg:.3e}, ", end="")
        print(f"EMA Critic Loss: {self.loss_critic_avg:.3e}")



class GSMRL(PPO):
  """GSMRL includes auxiliary info and intermediate reward from surrogate model."""
  def get_surrogate_model(self, config):
    self.use_surrogate = config["use_surrogate"]
    self.num_samples_auxiliary = config["num_samples_auxiliary"]
    surrogate_model_path = osp.join("experiments", "trained_models", config["dataset"], "acflow")
    surrogate_model_config = torch.load(osp.join(surrogate_model_path, f"config.pt"))
    self.surrogate_model = ACFlow(surrogate_model_config)
    repeat = config["repeat"] if "repeat" in config else 1
    self.surrogate_model.load(osp.join(surrogate_model_path, f"repeat_{repeat}"))
    self.surrogate_model.eval()
    for param in self.surrogate_model.parameters():
      param.requires_grad = False

  @torch.no_grad()
  def generate_auxiliary_info(self, x, mask):
    # Assumes that mask has already been multiplied by the missing data mask.
    # NOTE this is different to their paper, but matches the code.
    # We have opted to go with the public code, since this is both faster and is
    # more likely to be correct. The change is to not include the expected
    # I(Y; Xi | xo) term, which is in the paper but not the code. Instead the
    # auxiliary info is [p(y|xo), one_hot(y_pred), uncond_mean, uncond_std, cond_mean, cond_std].
    # Where uncond and cond refer to samples from p(xu|xo) and p(xu|xo, y_pred).
    # By including the standard deviations the expected information can
    # still be roughly extracted from this auxiliary information.
    self.surrogate_model.eval()
    self.predictive_model.eval()
    if self.use_surrogate:
      preds = self.surrogate_model.predict(x, mask)
    else:
      preds = self.predictive_model.predict(x, mask)
    y_pred = torch.argmax(preds, dim=-1)
    one_hot = F.one_hot(y_pred, num_classes=self.out_dim).float()

    batchsize = x.shape[0]
    unconditional_samples = self.surrogate_model.unconditional_samples(
      x, mask, torch.ones_like(mask), self.num_samples_auxiliary,
    ).reshape(batchsize, self.num_samples_auxiliary, -1)
    conditional_samples = self.surrogate_model.conditional_samples(
      x, y_pred, mask, torch.ones_like(mask), self.num_samples_auxiliary,
    ).reshape(batchsize, self.num_samples_auxiliary, -1)

    uncond_mean = torch.mean(unconditional_samples, dim=1)
    uncond_std = torch.std(unconditional_samples, dim=1)
    cond_mean = torch.mean(conditional_samples, dim=1)
    cond_std = torch.std(conditional_samples, dim=1)

    return torch.cat([preds, one_hot, uncond_mean, uncond_std, cond_mean, cond_std], dim=-1)

  @property
  def num_auxiliary_info(self):
    # The auxiliary info is p(y |x, m), the one-hot of the max of this,
    # and then mean and std deviations sampled from the p(x | xo) and p(x | xo, y).
    # num auxiliary info = 2*num_classes + 4*num_features
    return 2*self.out_dim + 4*self.num_features

  @torch.no_grad()
  def prepare_input(self, x, mask, auxiliary_info):
    return torch.cat([self.input_layer(x, mask), auxiliary_info], dim=-1)

  @torch.no_grad()
  def calculate_reward(self, x, y, auxiliary_info, m_curr, m_next):
    # Assumes that m_curr and m_next have ALREADY been multiplied by the
    # missing data mask.
    # NOTE we will likely make a change from the paper, to make GSMRL BETTER.
    # We have use_surrogate as a hyperparameter, the reward is calcualted at
    # every stop (prevent sparse reward). And it is the -cross entropy of the
    # predictions, add the information gain from the new feature. use_surrogate
    # tells us whether to use the surrogate model or the predictive model for
    # this. It is very likely it is best to use the predictive model.
    self.surrogate_model.eval()
    self.predictive_model.eval()
    if self.use_surrogate:
      logits_preds = self.surrogate_model.calc_logits(x, 0.0*m_next, m_next)
      logits_preds = logits_preds + self.surrogate_model.log_class_probs
      p0 = self.surrogate_model.predict(x, m_curr)
      p1 = self.surrogate_model.predict(x, m_next)
    else:
      logits_preds = self.predictive_model(x, m_next)
      p0 = self.predictive_model.predict(x, m_curr)
      p1 = self.predictive_model.predict(x, m_next)
    cross_entropy = F.cross_entropy(logits_preds, y, reduction="none")
    entropy0 = -torch.sum(p0*torch.log(p0 + log_eps), dim=-1)
    entropy1 = -torch.sum(p1*torch.log(p1 + log_eps), dim=-1)
    info_gain = entropy0 - entropy1
    return -cross_entropy + info_gain


