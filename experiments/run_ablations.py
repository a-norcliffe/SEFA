"""The code to launch ablations for SEFA.

Only a subset of the ablations require retraining SEFA:
  - Beta = 0
  - 1 Train Sample
  - Deterministic Encoder
  - WO copula
  - Sensitivity for beta
  - Sensitivity for number of train samples
  - Sensitivity for number of latents per feature

The other ablations affect how the acquisition is carried out, so do not need to
retrain the weights.
  - 1 Acq sample
  - Feature space calculation
  - WO normalization
  - WO probability weighting
  - Sensitivity for acquisition samples
"""


import os
import os.path as osp
import argparse

import numpy as np

import torch

from torch.utils.data import TensorDataset

from experiments.hyperparameters.utils import get_best_sweep_config
from experiments.metrics_dict import metrics_dict
from models.sefa import SEFA


parser = argparse.ArgumentParser()
parser.add_argument("--ablation", type=str, choices=["beta", "train_sample", "copula", "num_latents", "deterministic"])
parser.add_argument("--dataset", type=str)
parser.add_argument("--beta_value", type=str, default="0.0")  # We do it as string to maintain consistent folder saving names.
parser.add_argument("--num_latents", type=int, default=1)
parser.add_argument("--train_samples", type=int, default=1)
parser.add_argument("--first_run", type=int, default=1)
parser.add_argument("--last_run", type=int, default=5)
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()


# Set the device.
if args.device == "cpu":
  device = torch.device("cpu")
else:
  device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


# Load the data.
data_path = osp.join("datasets", "data", args.dataset)
dataset_dict = torch.load(osp.join(data_path, "dataset_dict.pt"))

dataset_x_ending = "std" if args.ablation == "copula" else "cdf"

X_train = torch.load(osp.join(data_path, f"X_train_{dataset_x_ending}.pt"))
y_train = torch.load(osp.join(data_path, f"y_train.pt"))
M_train = torch.load(osp.join(data_path, f"M_train.pt"))

X_val = torch.load(osp.join(data_path, f"X_val_{dataset_x_ending}.pt"))
y_val = torch.load(osp.join(data_path, f"y_val.pt"))
M_val = torch.load(osp.join(data_path, f"M_val.pt"))

train_data = TensorDataset(X_train, y_train, M_train)
val_data = TensorDataset(X_val, y_val, M_val)


# Create the path and directory.
folder_end = ""
if args.ablation == "beta":
  folder_end = f"_{args.beta_value.replace('.', '_')}"
if args.ablation == "train_sample":
  folder_end = f"_{str(args.train_samples)}"
if args.ablation == "num_latents":
  folder_end = f"_{str(args.num_latents)}"
save_path = osp.join("experiments", "trained_models", args.dataset, f"sefa_{args.ablation}{folder_end}")
os.makedirs(save_path, exist_ok=True)


# Get predictive metric function.
metric_f = metrics_dict[dataset_dict["metric"]]


# Construct the dictionary based on hyperparameters and dataset information.
config = get_best_sweep_config(args.dataset, "sefa")
for k, v in dataset_dict.items():
  config[k] = v

if args.ablation == "beta":
  config["beta"] = float(args.beta_value)
if args.ablation == "train_sample":
  config["num_samples_train"] = args.train_samples
if args.ablation == "num_latents":
  config["latent_dim"] = args.num_latents
torch.save(config, osp.join(save_path, f"config.pt"))



for repeat in range(args.first_run, args.last_run+1):
  rpt_path = osp.join(save_path, f"repeat_{repeat}")
  os.makedirs(rpt_path, exist_ok=True)

  print(f"\n\nRepeat {repeat} out of {args.last_run}")

  # Set the seed for consistency.
  seed = 8406*repeat + 383
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Setup and train the model.
  model = SEFA(config).to(device)
  if args.ablation == "deterministic":
    model.deterministic_encoding = True
  model.fit(train_data, val_data, rpt_path, metric_f)