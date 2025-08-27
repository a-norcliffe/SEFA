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
def get_acquisition_results(dataset_name, model_name):
  dataset_path = osp.join("datasets", "data", dataset_name)
  metric_f = metrics_dict[torch.load(osp.join(dataset_path, "dataset_dict.pt"))["metric"]]

  # Load in test data.
  data_x_ending = "cdf" if model_name == "sefa" else "std"
  X_test = torch.load(osp.join(dataset_path, f"X_test_{data_x_ending}.pt")).to(device)
  y_test = torch.load(osp.join(dataset_path, "y_test.pt")).to(device)
  M_test = torch.load(osp.join(dataset_path, "M_test.pt")).to(device)

  # Create the model.
  if model_name == "random":
    model_path = osp.join("experiments", "trained_models", dataset_name, "fixed_mlp")
  else:
    model_path = osp.join("experiments", "trained_models", dataset_name, model_name)
  model_config = torch.load(osp.join(model_path, "config.pt"))

  metric = []
  selected = []

  for repeat_num in range(1, NUM_REPEATS+1):
    print(f"Dataset: {dataset_name}, Model: {model_name}, Repeat: {repeat_num}/{NUM_REPEATS}")
    if model_name == "gsmrl":
      model_config["repeat"] = repeat_num
    model = models_dict[model_name](model_config).to(device)
    # This checks if the model finished training, if not there is an error.
    _ = torch.load(osp.join(model_path, f"repeat_{repeat_num}", "val_auc.pt"))
    model.load(osp.join(model_path, f"repeat_{repeat_num}"))
    model.eval()

    metric_tmp = []
    selected_tmp = []

    mask = torch.zeros_like(M_test)
    metric_tmp.append(metric_f(model.predict(X_test, mask*M_test), y_test))

    # Carry out the remaining acquisitions and store the selections.
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
  SEED = 9750
  np.random.seed(SEED)
  torch.manual_seed(SEED)

  # Create save folder.
  save_folder = osp.join("experiments", "results", "main")
  os.makedirs(save_folder, exist_ok=True)

  # NOTE: We can change this as we add models or if we only want to run a subset.
  models_list = list(models_dict.keys())
  models_list.remove("feature_space_ablation")

  # Run all the acquisitions in series. We parallelize by the dataset in screen sessions.
  results = {"metrics": {}, "selections": {}}

  for model_name in models_list:
    model_results = get_acquisition_results(args.dataset, model_name)
    results["metrics"][model_name] = model_results["metric"]
    results["selections"][model_name] = model_results["selected"]

    print("")

  # Save the results.
  torch.save(results, osp.join(save_folder, f"{args.dataset}.pt"))