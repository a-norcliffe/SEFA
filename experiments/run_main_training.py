"""The code to launch multiple training runs."""


import os
import os.path as osp
import argparse

import numpy as np
import torch

from torch.utils.data import TensorDataset

from experiments.hyperparameters.utils import get_best_sweep_config
from experiments.metrics_dict import metrics_dict
from models.models_dict import models_dict


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=list(models_dict.keys()))
parser.add_argument("--dataset", type=str)
parser.add_argument("--first_run", type=int, default=1)
parser.add_argument("--last_run", type=int, default=5)
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()


# Set the device.
if args.device == "cpu":
  device = torch.device("cpu")
else:
  device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


# Load the data, based on if it is our model or not.
data_path = osp.join("datasets", "data", args.dataset)
dataset_dict = torch.load(osp.join(data_path, "dataset_dict.pt"))

dataset_x_ending = "cdf" if args.model == "sefa" else "std"

X_train = torch.load(osp.join(data_path, f"X_train_{dataset_x_ending}.pt"))
y_train = torch.load(osp.join(data_path, f"y_train.pt"))
M_train = torch.load(osp.join(data_path, f"M_train.pt"))

X_val = torch.load(osp.join(data_path, f"X_val_{dataset_x_ending}.pt"))
y_val = torch.load(osp.join(data_path, f"y_val.pt"))
M_val = torch.load(osp.join(data_path, f"M_val.pt"))

train_data = TensorDataset(X_train, y_train, M_train)
val_data = TensorDataset(X_val, y_val, M_val)


# Create the path and directory.
save_path = osp.join("experiments", "trained_models", args.dataset, args.model)
os.makedirs(save_path, exist_ok=True)


# Get predictive metric function.
metric_f = metrics_dict[dataset_dict["metric"]]


# Construct the dictionary based on hyperparameters and dataset information.
config = get_best_sweep_config(args.dataset, args.model)
for k, v in dataset_dict.items():
  config[k] = v
torch.save(config, osp.join(save_path, f"config.pt"))

for repeat in range(args.first_run, args.last_run+1):
  # For GSMRL to load in an MLP and ACFlow model.
  if args.model == "gsmrl":
    config["repeat"] = repeat

  rpt_path = osp.join(save_path, f"repeat_{repeat}")
  os.makedirs(rpt_path, exist_ok=True)

  print(f"\n\nRepeat {repeat} out of {args.last_run}")

  # Set the seed for consistency.
  seed = 8406*repeat + 383
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Setup and train the model.
  model = models_dict[args.model](config).to(device)
  model.fit(train_data, val_data, rpt_path, metric_f)