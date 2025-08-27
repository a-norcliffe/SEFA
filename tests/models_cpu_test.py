"""Testing that models load properly, i.e. they produce the same results
after loading. And checking results don't change when changing masked values,
and check the results do change when changing unmasked values.

All tested on CPU, there is a separate GPU test.
"""


import os
import os.path as osp
import unittest

import numpy as np

import torch

from experiments.hyperparameters.utils import get_model_config
from models.models_dict import models_dict


device = torch.device("cpu")
batchsize = 10
most_categories = 3

# Skip models that have additional loading procedures. Feature space ablation
# is built on VAE, so if VAE works the feature space ablation works.
# Random is built on fixed mlp, so if fixed mlp works so does random.
models_no_test = ["feature_space_ablation", "random"]

dataset_dict = {
  "dataset": "testing_dataset",
  "most_categories": most_categories,
  "out_dim": 2,
  "metric": "auroc",
  "max_dim": None,
  "log_class_probs": torch.tensor([np.log(0.5), np.log(0.5)]),
}


def make_features_mask(num_con_features, num_cat_features, most_categories):
  x_con = torch.randn((batchsize, num_con_features))
  x_cat = torch.randint(high=most_categories, size=(batchsize, num_cat_features)).float()
  x = torch.cat([x_con, x_cat], dim=1).to(device)
  m = torch.bernoulli(0.5*torch.ones_like(x))  # Has same device as x.
  return x, m


def perturb_features(x, num_con_features, num_cat_features, most_categories):
  if num_con_features > 0:
    x[:, 0] = -x[:, 0]
  else:
    x[:, 0] = ((x[:, 0] + 1.0) % most_categories).float()
  if num_cat_features > 0:
    x[:, -1] = ((x[:, -1] + 1.0) % most_categories).float()
  else:
    x[:, -1] = -x[:, -1]
  return x


def remove_temp_files():
  try:
    os.remove("best_model.pt")
  except FileNotFoundError:
    pass
  try:
    os.remove("fixed_order_scores.pt")
  except FileNotFoundError:
    pass
  try:
    delete_gsmrl_test_files()
  except:
    pass


def make_gsmrl_pretrained_files(dataset_config, config_num):
  path = osp.join("experiments", "trained_models", "testing_dataset")
  acflow_path = osp.join(path, "acflow")
  fixed_mlp_path = osp.join(path, "fixed_mlp")

  os.makedirs(osp.join(acflow_path, "repeat_1"), exist_ok=True)
  os.makedirs(osp.join(fixed_mlp_path, "repeat_1"), exist_ok=True)

  acflow_config = get_model_config("acflow", config_num)
  fixed_mlp_config = get_model_config("fixed_mlp", config_num)

  for k, v in dataset_config.items():
    acflow_config[k] = v
    fixed_mlp_config[k] = v

  torch.save(acflow_config, osp.join(acflow_path, "config.pt"))
  torch.save(fixed_mlp_config, osp.join(fixed_mlp_path, "config.pt"))

  acflow_path = osp.join(acflow_path, "repeat_1")
  fixed_mlp_path = osp.join(fixed_mlp_path, "repeat_1")

  acflow_model = models_dict["acflow"](acflow_config)
  torch.save(acflow_model.state_dict(), osp.join(acflow_path, "best_model.pt"))

  fixed_mlp_model = models_dict["fixed_mlp"](fixed_mlp_config)
  fixed_order_scores = torch.arange(dataset_config["num_con_features"] + dataset_config["num_cat_features"]).float()
  torch.save(fixed_mlp_model.state_dict(), osp.join(fixed_mlp_path, "best_model.pt"))
  torch.save(fixed_order_scores, osp.join(fixed_mlp_path, "fixed_order_scores.pt"))


def delete_gsmrl_test_files():
  tmp_path = osp.join("experiments", "trained_models", "testing_dataset")
  for root, dirs, files in os.walk(tmp_path, topdown=False):
    for file in files:
      os.remove(osp.join(root, file))
    for dir in dirs:
      os.rmdir(osp.join(root, dir))
  os.rmdir(tmp_path)


class TestModels(unittest.TestCase):

  @torch.no_grad()
  def test_loading(self):
    for model_name in models_dict.keys():
      if model_name in models_no_test:
        continue

      for config_num in range(1, 9+1):
        config = get_model_config(model_name, config_num)

        for num_con_features, num_cat_features in zip([5, 5, 0], [5, 0, 5]):
          x, m = make_features_mask(num_con_features, num_cat_features, most_categories)

          dataset_dict["num_con_features"] = num_con_features
          dataset_dict["num_cat_features"] = num_cat_features
          for k, v in dataset_dict.items():
            config[k] = v

          if model_name == "gsmrl":
            make_gsmrl_pretrained_files(dataset_dict, config_num)

          model1 = models_dict[model_name](config).to(device)
          model1.eval()
          torch.save(model1.state_dict(), "best_model.pt")
          if model_name == "fixed_mlp":
            model1.fixed_order_scores = torch.arange(num_con_features + num_cat_features, device=device).float()
            torch.save(model1.fixed_order_scores, "fixed_order_scores.pt")
          model2 = models_dict[model_name](config).to(device)
          model2.eval()
          model2.load("")

          torch.manual_seed(42)
          py1 = model1.predict(x, m)
          scores1 = model1.calculate_acquisition_scores(x, m)

          torch.manual_seed(42)
          py2 = model2.predict(x, m)
          scores2 = model2.calculate_acquisition_scores(x, m)

          assert torch.all(py1 == py2), f"{model_name} failed at loading"
          assert torch.all(scores1 == scores2), f"{model_name} failed at loading"
          remove_temp_files()

  @torch.no_grad()
  def test_masking(self):
    for model_name in models_dict.keys():
      if model_name in models_no_test:
        continue

      for config_num in range(1, 9+1):
        config = get_model_config(model_name, config_num)

        for num_con_features, num_cat_features in zip([5, 5, 0], [5, 0, 5]):
          x1, m = make_features_mask(num_con_features, num_cat_features, most_categories)
          x2 = perturb_features(x1.clone(), num_con_features, num_cat_features, most_categories)

          dataset_dict["num_con_features"] = num_con_features
          dataset_dict["num_cat_features"] = num_cat_features
          for k, v in dataset_dict.items():
            config[k] = v

          if model_name == "gsmrl":
            make_gsmrl_pretrained_files(dataset_dict, config_num)

          model = models_dict[model_name](config).to(device)
          model.eval()
          if model_name == "fixed_mlp":
            model.fixed_order_scores = torch.arange(num_con_features + num_cat_features, device=device).float()

          # Mask value = 0.0
          m[:, 0] = 0.0
          m[:, -1] = 0.0

          torch.manual_seed(42)
          py1 = model.predict(x1, m)
          scores1 = model.calculate_acquisition_scores(x1, m)

          torch.manual_seed(42)
          py2 = model.predict(x2, m)
          scores2 = model.calculate_acquisition_scores(x2, m)

          assert torch.all(py1 == py2), f"{model_name} config {config_num} failed at masking 0"
          assert torch.all(scores1 == scores2), f"{model_name} config {config_num} failed at masking 0"

          # Mask value = 1.0
          m[:, 0] = 1.0
          m[:, -1] = 1.0

          torch.manual_seed(42)
          py1 = model.predict(x1, m)
          scores1 = model.calculate_acquisition_scores(x1, m)

          torch.manual_seed(42)
          py2 = model.predict(x2, m)
          scores2 = model.calculate_acquisition_scores(x2, m)

          assert not torch.allclose(py1, py2, atol=0.0001), f"{model_name} config {config_num} failed at masking 1"
          if model_name != "fixed_mlp":
            assert not torch.allclose(scores1, scores2, atol=0.0001), f"{model_name} config {config_num} failed at masking 1"
          remove_temp_files()


if __name__ == "__main__":
  print(device)
  unittest.main()