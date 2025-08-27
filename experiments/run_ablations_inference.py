import os
import os.path as osp

import argparse

import numpy as np
import torch

from experiments.metrics_dict import metrics_dict
from models.models_dict import models_dict


NUM_REPEATS = 5


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()



# Set the device.
if args.device == "cpu":
  device = torch.device("cpu")
else:
  device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def get_acquisition_results(dataset_name, ablation_name):
  dataset_path = osp.join("datasets", "data", dataset_name)
  metric_f = metrics_dict[torch.load(osp.join(dataset_path, "dataset_dict.pt"))["metric"]]

  # Load in test data.
  if (ablation_name == "copula" or ablation_name == "feature_space_ablation"):
    data_x_ending = "std"
  else:
    data_x_ending = "cdf"
  X_test = torch.load(osp.join(dataset_path, f"X_test_{data_x_ending}.pt")).to(device)
  y_test = torch.load(osp.join(dataset_path, "y_test.pt")).to(device)
  M_test = torch.load(osp.join(dataset_path, "M_test.pt")).to(device)

  # Get model name.
  if ablation_name == "feature_space_ablation":
    model_name = "feature_space_ablation"
  else:
    model_name = "sefa"

  # Get model folder.
  if (ablation_name == "copula" or ablation_name == "deterministic"):
    model_folder = f"sefa_{ablation_name}"
  elif ablation_name == "train_sample":
    model_folder = "sefa_train_sample_1"
  elif ablation_name == "beta":
    model_folder = "sefa_beta_0_0"
  elif ablation_name == "num_latents":
    model_folder = "sefa_num_latents_1"
  elif ablation_name == "feature_space_ablation":
    model_folder = "vae"
  else:
    model_folder = "sefa"
  model_path = osp.join("experiments", "trained_models", dataset_name, model_folder)
  model = models_dict[model_name](torch.load(osp.join(model_path, "config.pt"))).to(device)

  if ablation_name == "acq_sample":
    model.num_samples_acquire = 1
  if ablation_name == "deterministic":
    model.deterministic_encoding = True
  if ablation_name == "no_normalize":
    model.normalize_grads = False
  if ablation_name == "prob_weighting":
    model.prob_weighting = False

  metric = []
  selected = []

  for repeat_num in range(1, NUM_REPEATS+1):
    print(f"Dataset: {dataset_name}, Ablation: {ablation_name}, Repeat: {repeat_num}/{NUM_REPEATS}")
    # This checks if the model finished training, if not there is an error.
    _ = torch.load(osp.join(model_path, f"repeat_{repeat_num}", "val_auc.pt"))
    model.load(osp.join(model_path, f"repeat_{repeat_num}"))
    model.eval()

    metric_tmp = []
    selected_tmp = []

    mask = torch.zeros_like(M_test)
    metric_tmp.append(metric_f(model.predict(X_test, mask*M_test), y_test))

    # Carry out the remaining acquisitions.
    for _ in range(M_test.shape[-1]):
      mask, selection = model.acquire(x=X_test, mask_acq=mask, mask_data=M_test, return_features=True)
      metric_tmp.append(metric_f(model.predict(X_test, mask*M_test), y_test))
      selected_tmp.append(selection.cpu())

    # Append to repeated arrays.
    metric.append(torch.tensor(metric_tmp))
    selected.append(torch.stack(selected_tmp, dim=0))

  # Stack across the repeats and return.
  metric = torch.stack(metric, dim=0).float()
  selected = torch.stack(selected, dim=0).long()

  return {
    "metric": metric,
    "selected": selected,
  }



if __name__ == "__main__":
  # Set a seed for reproducibility.
  SEED = 1852
  np.random.seed(SEED)
  torch.manual_seed(SEED)

  # Create save folder.
  save_folder = osp.join("experiments", "results", "ablations")
  os.makedirs(save_folder, exist_ok=True)

  # NOTE we can add to this if we want to.
  ablations_list = [
    "train_sample",
    "copula",
    "beta",
    "acq_sample",
    "deterministic",
    "num_latents",
    "feature_space_ablation",
    "no_normalize",
    "prob_weighting",
  ]

  results = {"metrics": {}, "selections": {}}

  for ablation_name in ablations_list:
    ablation_results = get_acquisition_results(args.dataset, ablation_name)
    results["metrics"][ablation_name] = ablation_results["metric"]
    results["selections"][ablation_name] = ablation_results["selected"]

    print("")

  # Save the results.
  torch.save(results, osp.join(save_folder, f"{args.dataset}.pt"))