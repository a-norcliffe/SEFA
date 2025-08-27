"""Data available at
https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification
The data is from the MiniBooNE experiment at Fermilab.
Labels are 1 for signal, 0 for background. The positive signal is a muon
neutrino oscillating into an electron neutrino.
The data is a binary classification problem, with 130,064 examples and 50 features.
First 36,499 are positive examples, the remaining 93,565 are negative.

We use 40,000 negative examples to enforce class balance, so the dataset size
is 76,499. We use val and test size = 10,000 so train size is 56,499.
"""


import sys
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch

from datasets.preprocessing_utils import preprocess_and_save_data


# Set seed.
SEED = 4903
np.random.seed(SEED)
torch.manual_seed(SEED)

path = osp.join("datasets", "data", "miniboone")
os.makedirs(path, exist_ok=True)

val_size = 10_000
test_size = 10_000


if __name__ == "__main__":
  # Load in miniboone raw csv.
  try:
    print("Loading raw data.")
    X = pd.read_csv(osp.join(path, "MiniBooNE_PID.txt"), sep="\s+", skiprows=1, header=None).to_numpy()
    num_pos_neg = pd.read_csv(osp.join(path, "MiniBooNE_PID.txt"), sep="\s+", nrows=1, header=None).to_numpy()
    print("Data loaded.")
  except FileNotFoundError:
    print("Please download the miniboone data from https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification")
    print("and place it in the datasets/data/miniboone folder.")
    print("Or use the Linux commands (worked on 08/08/2025):")
    print("> wget -P datasets/data/miniboone https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip")
    print("> unzip datasets/data/miniboone/miniboone+particle+identification -d datasets/data/miniboone")
    print("> rm datasets/data/miniboone/miniboone+particle+identification.zip")
    sys.exit()

  # Find how many positive and how many negative.
  num_positive = num_pos_neg[0, 0]
  num_negative = num_pos_neg[0, 1]

  # Separate positive and negative examples, shuffle negative and select roughly
  # same number as positive.
  X_pos = X[:num_positive]
  X_neg = X[num_positive:]
  shuffle_ids = np.random.permutation(X_neg.shape[0])
  X_neg = X_neg[shuffle_ids]
  X_neg = X_neg[:round(num_positive, -4)]  # Round to the nearest 10000 (40000).

  # Construct the X and y data and convert to tensors.
  X = np.concatenate([X_pos, X_neg], axis=0)
  y = np.zeros(shape=(X.shape[0]))
  y[:num_positive] = 1.0

  X = torch.tensor(X).float()
  y = torch.tensor(y).long()

  # Extract the best features based on XGBoost feature importances.
  best_features = np.array([2, 3, 6, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29,
                            34, 39, 40, 41, 42, 43, 44])
  X = X[:, best_features]

  dataset_dict = {
    "dataset": "miniboone",
    "num_con_features": X.shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": 2,
    "metric": "auroc",
    "max_dim": None,
  }

  preprocess_and_save_data(
    path=path,
    dataset_dict=dataset_dict,
    train_size=X.shape[0] - (val_size + test_size),
    val_size=val_size,
    X=X,
    y=y,
    M=None,
    shuffle=True,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.05,
  )




