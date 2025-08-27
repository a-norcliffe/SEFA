"""Implementation of Opportunistic Learning RL method,
Paper: https://arxiv.org/abs/1901.00243
Code: https://github.com/mkachuee/Opportunistic/tree/master
Implementation follows the original code as closely as possible, as given in
their notebook: https://github.com/mkachuee/Opportunistic/blob/master/Demo_OL_DQN.ipynb

Some parts are simplified for our purposes. For example we do not train with a
stream of data, we allow the model to see batches every episode. We also do not
have an end acquisition action.
The other main thing we change in line with
the traditional implementation of Deep Q Learning (
  - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html,
  - https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm
  - https://www.nature.com/articles/nature14236
)
is to let the policy network decide the actions, rather than the target network,
and let the q target value be r + gamma * max_a'(Q_T(s1, a')), rather than
r + gamma * Q_T(s1, a') where a' = argmax_a(Q(s1, a)). These are both changes
to the OL implementation.
"""

import os.path as osp

import numpy as np
from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader

from models.base import BaseModel
from models.rl_utils import IterableIds, ExperienceBuffer


class PQNetwork(nn.Module):
  """The PQ network where hidden representations from the P network are 
  used by the Q network. Therefore we assume num_hidden > 0. NOTE: the
  implementation is not built for num_hidden=0.
  """
  def __init__(self, input_layer, in_dim, hidden_dim, p_out_dim, q_out_dim, num_hidden):
    super().__init__()
    self.input_layer = input_layer

    # Set up the linear layers, first layers initially.
    self.p_linears = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
    self.q_linears = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
    # Intermediate layers.
    for _ in range(num_hidden-1):
      self.p_linears.append(nn.Linear(hidden_dim, hidden_dim))
      self.q_linears.append(nn.Linear(2*hidden_dim, hidden_dim))
    # Final linear layers.
    self.p_linears.append(nn.Linear(hidden_dim, p_out_dim))
    self.q_linears.append(nn.Linear(2*hidden_dim, q_out_dim))

  def forward(self, x, mask):
    x = self.input_layer(x, mask)
    # First layers.
    p_out = self.p_linears[0](x)
    q_out = self.q_linears[0](x)
    # Subsequent layers.
    for p_linear, q_linear in zip(self.p_linears[1:], self.q_linears[1:]):
      p_out = F.dropout(F.relu(p_out), p=0.5, training=True)
      q_out = q_linear(torch.cat([F.relu(q_out), p_out.detach()], dim=-1))
      p_out = p_linear(p_out)
    return p_out, F.softplus(q_out)  # Q values are always > 0.

  def p_only(self, x, mask):
    p_out = self.p_linears[0](self.input_layer(x, mask))
    for p_linear in self.p_linears[1:]:
      p_out = p_linear(F.dropout(F.relu(p_out), p=0.5, training=True))
    return p_out

  def q_only(self, x, mask):
    return self.forward(x, mask)[1]

  def predict(self, x, mask, num_samples):
    x = torch.repeat_interleave(x, num_samples, dim=0)
    mask = torch.repeat_interleave(mask, num_samples, dim=0)
    out = F.log_softmax(self.p_only(x, mask), dim=-1)
    out = out.view(-1, num_samples, out.shape[-1])
    out = torch.logsumexp(out, dim=1) - np.log(num_samples)
    out = torch.exp(out)
    return out

  def update(self, update_model, update_rate=1.0):
    for self_param, update_param in zip(self.parameters(), update_model.parameters()):
      self_param.data.copy_(self_param.data*(1.0 - update_rate) + update_param.data*update_rate)


class OpportunisticRL(BaseModel):
  """Opportunistic Learning RL model. This can be used directly, the
  initialization handles the input layers.
  """
  def __init__(self, config):
    super().__init__(config)
    self.gamma = config["gamma"]
    self.num_samples_predict = config["num_samples_predict"]
    self.num_episodes = config["num_episodes"]
    self.iter_per_exp = 1 + self.num_features//100

    self.pq_net_kwargs = {
      "input_layer": self.input_layer,
      "in_dim": self.in_dim,
      "hidden_dim": config["hidden_dim"],
      "p_out_dim": self.out_dim,
      "q_out_dim": self.num_features,
      "num_hidden": config["num_hidden"],
    }
    self.model = PQNetwork(**self.pq_net_kwargs)

    # Parameters for printing and regular evaluation.
    self.print_every = 500
    self.eval_every = 100
    self.ema_alpha = 0.2

  def predict(self, x, mask):
    return self.model.predict(x, mask, self.num_samples_predict)

  @torch.no_grad()
  def calculate_acquisition_scores(self, x, mask):
    return self.model.q_only(x, mask)

  def fit_parameters(self, train_data, val_data, save_path, metric_f):
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    optimizer = Adam(self.parameters(), lr=self.lr)
    target_model = PQNetwork(**self.pq_net_kwargs).to(self.device)
    target_model.update(self.model, update_rate=1.0)
    buffer = ExperienceBuffer(buffer_size=self.num_features*1000)
    train_ids_iter = IterableIds(len(train_data), self.batchsize)

    # Start training.
    prob_rand = 1.0
    experience_count = 0
    best_val_auc = 0

    print("Starting training from scratch\n")
    for episode in range(1, self.num_episodes+1):
      self.train()

      # After 10% of episodes decay random action probability every episode.
      if episode > int(0.1*self.num_episodes):
        # Decay random action probability.
        prob_rand = max(0.1, prob_rand*(0.1**(1/self.num_episodes)))

      # Decay learning rate if we are at 0.5, 0.6, 0.7, 0.8, 0.9 through training.
      if episode in [int(f*self.num_episodes) for f in [0.5, 0.6, 0.7, 0.8, 0.9]]:
        print(f"Decaying learning rate from {self.lr:.3e} to {self.lr*0.2:.3e}")
        self.lr *= 0.2
        optimizer = Adam(self.parameters(), lr=self.lr)

      # Get new sample and play out the episode.
      # We can train offline since we have the dataset, rather than the online
      # setting as originally implemented in OL.
      random_ids = train_ids_iter.next()
      x_ep, y_ep, m_data_ep = train_data[random_ids]
      x_ep = x_ep.to(self.device)
      y_ep = y_ep.to(self.device)
      m_data_ep = m_data_ep.to(self.device)

      with torch.no_grad():
        m_curr = torch.zeros_like(m_data_ep)
        p_curr = self.predict(x_ep, m_curr*m_data_ep)  # Predict y dist from no mask.

      for _ in range(self.max_dim):
        with torch.no_grad():
          experience_count += 1
          if np.random.random() < prob_rand:
            # Sample random but allowed action.
            scores = torch.rand_like(m_curr)
          else:
            # Choose action from policy Q network, NOTE: Different from OL code.
            scores = self.model.q_only(x_ep, m_curr*m_data_ep)
          # Carry out same score update as in our base model to choose features
          # that have not been selected and are available. Or if nothing that
          # has not been selected is available we simply choose the first that 
          # has not been selected yet.
          scores += 1.0
          scores *= (1-m_curr)*m_data_ep
          scores += (1-m_curr)*1e-6
          selected = F.one_hot(torch.argmax(scores, dim=-1), num_classes=self.num_features).float()
          m_next = torch.max(m_curr, selected)

          p_next = self.predict(x_ep, m_next*m_data_ep) # Predict y dist from new mask.
          reward = torch.sum(torch.abs(p_next - p_curr), dim=-1)

          # Push the experience in, which includes multiplcation by data mask to
          # show if they were available actions.
          buffer.append(x_ep, y_ep, m_curr*m_data_ep, m_next*m_data_ep, reward)
          p_curr = p_next
          m_curr = m_next

          # Check if we are ready to carry out gradient updates.
          if ((buffer.length<self.batchsize) or (experience_count%self.iter_per_exp!=0)):
            continue

        with torch.enable_grad():
          x, y, m0, m1, r = buffer.sample(self.batchsize)

          # We always update the predictor.
          optimizer.zero_grad()
          y_logits = self.model.p_only(x, m0)
          loss_p = F.cross_entropy(y_logits, y)
          loss_p.backward()
          optimizer.step()
          if hasattr(self, "loss_p_avg"):
            self.loss_p_avg = self.loss_p_avg*(1-self.ema_alpha) + loss_p.item()*self.ema_alpha
          else:
            self.loss_p_avg = loss_p.item()

          # After 10% of training we train the Q network as well as P network.
          if episode >= int(0.1*self.num_episodes):
            optimizer.zero_grad()

            # What are the q values of the actions taken specifically.
            q_values = torch.sum(self.model.q_only(x, m0)*(m1 - m0), dim=-1)

            # Select best action from the target model. NOTE: Different to OL code.
            with torch.no_grad():
              q_target_values = torch.max(target_model.q_only(x, m1), dim=-1)[0]
              q_target_values = (r + self.gamma*q_target_values)

            loss_q = torch.mean((q_values - q_target_values)**2)
            loss_q.backward()
            optimizer.step()
            if hasattr(self, "loss_q_avg"):
              self.loss_q_avg = self.loss_q_avg*(1-self.ema_alpha) + loss_q.item()*self.ema_alpha
            else:
              self.loss_q_avg = loss_q.item()

        # Update target model.
        target_model.update(self.model, update_rate=0.001)

      # Evaluate validation info and update exponential moving averages.
      # And checkpoint the model.
      if episode%self.eval_every == 0:
        self.eval()
        val_metric, _ = self.calc_val_dict(val_loader, metric_f)
        val_auc = self.run_zero_acquisition(val_loader, metric_f)
        # Set the first value of the average. Otherwise exponentially moving average.
        if hasattr(self, "val_metric_avg"):
          self.val_metric_avg = self.val_metric_avg*(1-self.ema_alpha) + val_metric*self.ema_alpha
          self.val_auc_avg = self.val_auc_avg*(1-self.ema_alpha) + val_auc*self.ema_alpha
        else:
          self.val_metric_avg = val_metric
          self.val_auc_avg = val_auc

        if val_auc > best_val_auc:
          best_val_auc = val_auc
          torch.save(self.state_dict(), osp.join(save_path, "best_model.pt"))

      # Print information about episodes.
      if episode%self.print_every == 0:
        print(f"Episode {episode}/{self.num_episodes}, ", end="")
        print(f"EMA P Loss: {self.loss_p_avg:.3e}, ", end="")
        if hasattr(self, "loss_q_avg"):
          print(f"EMA Q Loss: {self.loss_q_avg:.3e}, ", end="")
        print(f"EMA Val Metric: {self.val_metric_avg:.3f}, ", end="")
        print(f"EMA Val AUC: {self.val_auc_avg:.3f}, ", end="")
        print(f"Best Val AUC: {best_val_auc:.3f}")

