"""Function to get model configs. For a given model and given config number."""


import os
import os.path as osp

import json

import torch


def get_model_config(model_name, config_num):
  path = osp.join("experiments", "hyperparameters", "configs", f"{model_name}", f"config{config_num}.json")
  with open(path, "r") as f:
    config = json.load(f)
  return config


def get_best_sweep_config(dataset, model_name):
  sweep_results_path = osp.join("experiments", "hyperparameters", "tuning_results", dataset, model_name)
  best_config = 0
  best_performance = -1e6
  for file in os.listdir(sweep_results_path):
    config_num = int(file[len("config"):-len(".pt")])
    config_performance = torch.load(osp.join(sweep_results_path, file))
    # We want the best mean performance, if it is too close to call, we minus
    # a small amount of standard deviation.
    config_performance = config_performance["mean"] - 1e-4*config_performance["std"]

    if config_performance > best_performance:
      best_performance = config_performance
      best_config = config_num
  return get_model_config(model_name, best_config)