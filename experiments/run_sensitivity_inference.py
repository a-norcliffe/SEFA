import os
import os.path as osp

import argparse

import numpy as np

import torch

from models.sefa import SEFA


NUM_REPEATS = 5


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["syn1", "syn2", "syn3"])
parser.add_argument("--ablation", type=str, choices=["beta", "train_sample", "acq_sample", "num_latents"])
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()



# Set the device.
if args.device == "cpu":
  device = torch.device("cpu")
else:
  device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")



@torch.no_grad()
def get_acquisition_array(dataset_name, sensitivity_value, ablation):
  dataset_path = osp.join("datasets", "data", dataset_name)

  # Load in test data.
  X_test = torch.load(osp.join(dataset_path, "X_test_cdf.pt")).to(device)
  M_test = torch.load(osp.join(dataset_path, "M_test.pt")).to(device)

  # Create the model.
  if ablation == "acq_sample":
    model_path = osp.join("experiments", "trained_models", dataset_name, f"sefa")
  else:
    model_path = osp.join("experiments", "trained_models", dataset_name, f"sefa_{ablation}_{sensitivity_value}")
  config = torch.load(osp.join(model_path, "config.pt"))
  model = SEFA(config).to(device)
  if ablation == "acq_sample":
    model.num_samples_acquire = sensitivity_value

  selected = []

  for repeat_num in range(1, NUM_REPEATS+1):
    print(f"Dataset: {dataset_name}, {ablation}: {sensitivity_value}, Repeat: {repeat_num}/{NUM_REPEATS}")
    # Check if the model finished training.
    _ = torch.load(osp.join(model_path, f"repeat_{repeat_num}", "val_auc.pt"))
    model.load(osp.join(model_path, f"repeat_{repeat_num}"))
    model.eval()

    selected_tmp = []
    mask = torch.zeros_like(M_test)

    # Carry out the remaining acquisitions actively.
    for _ in range(M_test.shape[-1]):
      mask, selection = model.acquire(x=X_test, mask_acq=mask, mask_data=M_test, return_features=True)
      selected_tmp.append(selection.cpu())

    selected.append(torch.stack(selected_tmp, dim=0))

  selected = torch.stack(selected, dim=0).long()

  return selected



if __name__ == "__main__":
  # Set the seed for consistency.
  seed = 1945
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Create save folder.
  save_folder = osp.join("experiments", "results", "ablations", "sensitivity", args.dataset)
  os.makedirs(save_folder, exist_ok=True)

  results = {}

  # Run all the acquisitions in series. We parallelize by the dataset and ablation in screen sessions.
  # We will include beta=0.0, 1 acq sample, 1 train sample, latent dim=1,
  # in the standard ablations, and load them in the evaluation notebook.
  if args.ablation == "beta":
    sensitivity_values = ["0_000001", "0_00001", "0_0001", "0_001", "0_01", "0_1", "1_0", "10_0"]
  elif args.ablation == "train_sample":
    sensitivity_values = [3, 10, 30, 100, 300, 1000, 3000, 10000]
  elif args.ablation == "acq_sample":
    sensitivity_values = [3, 10, 30, 100, 300, 1000, 3000, 10000]
  elif args.ablation == "num_latents":
    sensitivity_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
  else:
    raise ValueError("Ablation not implemented.")

  for x in sensitivity_values:
    results[x] = get_acquisition_array(args.dataset, x, args.ablation)
    print("")

  # Save the results.
  torch.save(results, osp.join(save_folder, f"{args.ablation}.pt"))