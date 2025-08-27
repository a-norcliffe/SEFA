"""The code to launch hyperparameter sweeps."""


import os
import os.path as osp
import argparse

import numpy as np

import torch

from torch.utils.data import TensorDataset

from experiments.metrics_dict import metrics_dict
from experiments.hyperparameters.utils import get_model_config
from models.models_dict import models_dict


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=list(models_dict.keys()))
parser.add_argument("--dataset", type=str)
parser.add_argument("--first_config", type=int, default=1)
parser.add_argument("--last_config", type=int, default=9)
parser.add_argument("--num_repeats", type=int, default=3)
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()


# Set the device.
if args.device == "cpu":
  device = torch.device("cpu")
else:
  device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


# Load the data, based on if it is SEFA or not.
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


# Get predictive metric function.
metric_f = metrics_dict[dataset_dict["metric"]]


# Construct the dictionary based on hyperparameters and dataset information.
def get_aucs(config_num):
  config = get_model_config(args.model, config_num)
  # For GSMRL to load in an MLP and ACFlow model.
  if args.model == "gsmrl":
    config["repeat"] = 1
  for key in dataset_dict.keys():
    config[key] = dataset_dict[key]

  aucs = []
  for repeat in range(1, args.num_repeats+1):
    print(f"\n\nRepeat {repeat} out of {args.num_repeats}")

    save_path = osp.join("experiments", "hyperparameters", "tuning_tmp", args.dataset, args.model, f"config{config_num}", f"repeat_{repeat}")
    os.makedirs(save_path, exist_ok=True)

    # Set the seed for consistency.
    seed = 1690*repeat + 241
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup and train the model.
    model = models_dict[args.model](config).to(device)
    model.fit(train_data, val_data, save_path, metric_f)
    aucs.append(torch.load(osp.join(save_path, "val_auc.pt")))

  aucs = np.array(aucs)
  return aucs


# Create dictionary of aucs, this is the random search.
result_path = osp.join("experiments", "hyperparameters", "tuning_results", args.dataset, args.model)
os.makedirs(result_path, exist_ok=True)
for config_num in range(args.first_config, args.last_config + 1):
  aucs = get_aucs(config_num)
  tuning_result = {"mean": np.mean(aucs), "std": np.std(aucs), "aucs": aucs}
  torch.save(tuning_result, osp.join(result_path, f"config{config_num}.pt"))

# Delete folders used in the sweeping. Do it after, so we get no errors before
# saving the sweep results.
for config_num in range(args.first_config, args.last_config + 1):
  delete_path = osp.join("experiments", "hyperparameters", "tuning_tmp", args.dataset, args.model, f"config{config_num}")
  for root, dirs, files in os.walk(delete_path, topdown=False):
    for file in files:
      os.remove(osp.join(root, file))
    for dir in dirs:
      os.rmdir(osp.join(root, dir))
  os.rmdir(delete_path)