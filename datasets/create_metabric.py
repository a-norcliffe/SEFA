"""METABRIC dataset from https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric
The dataset contains gene expression profiles of
breast cancer patients. The task is to predict the PAM50 status using
the gene expressions.

The classes are:
  1. LumA
  2. LumB
  3. Her2
  4. claudin-low
  5. Basal
  6. Normal

The 12 features selected by STG are:
  1. ccnb1
  2. cdk1
  3. e2f2
  4. e2f7
  5. stat5b
  6. notch1
  7. rbpj
  8. bcl2
  9. egfr
  10. erbb2
  11. erbb3
  12. abcb1
"""


import sys
import os
import os.path as osp

import numpy as np
import pandas as pd

import torch

from datasets.preprocessing_utils import preprocess_and_save_data


# Set seed.
SEED = 2534
np.random.seed(SEED)
torch.manual_seed(SEED)

path = osp.join("datasets", "data", "metabric")
os.makedirs(path, exist_ok=True)

train_ratio = 0.8
val_ratio = 0.1


if __name__ == "__main__":
  try:
    df = pd.read_csv(osp.join(path, "METABRIC_RNA_Mutation.csv"), low_memory=False)
  except FileNotFoundError:
    print("Please download the METABRIC data from https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric")
    print("and place METABRIC_RNA_Mutation.csv in the datasets/data/metabric folder.")
    print("There are no easy Linux commands, (disregarding the Kaggle API).")
    sys.exit()

  # Remove missing labels.
  missing_label_indices = df[ df["pam50_+_claudin-low_subtype"] == "NC" ].index
  df.drop(missing_label_indices, inplace = True)

  df_x = df.iloc[:, 31:]
  z_columns = []  # Continuous z value of gene (x - mu)/sigma.
  mut_columns = []  # Categorical mutation of gene.
  for col in df_x.columns:
    if col.endswith("_mut"):
      mut_columns.append(col)
    else:
      z_columns.append(col)

  df_z = df_x[z_columns]
  df_muts = df_x[mut_columns]
  df_muts = (df_muts != "0").astype(float)

  X = np.concatenate([df_z.values, df_muts.values], axis=1)

  mapping_dict = {
    "LumA": 0,
    "LumB": 1,
    "Her2": 2,
    "claudin-low": 3,
    "Basal": 4,
    "Normal": 5,
  }
  y = np.array([mapping_dict[i] for i in df["pam50_+_claudin-low_subtype"]])

  # Features selected by STG.
  # All selected featres are continuous since they are z values.
  # Mutation features begin after 489. Max here is 290.
  best_features = np.array([24, 25, 39, 44, 53, 89, 98, 122, 152, 156, 157, 290])

  X = X[:, best_features]
  X = torch.tensor(X).float()
  y = torch.tensor(y).long()

  dataset_dict = {
    "dataset": "metabric",
    "num_con_features": X.shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": int(y.max().item()+1),
    "metric": "accuracy",
    "max_dim": None,
  }

  preprocess_and_save_data(
    path=path,
    dataset_dict=dataset_dict,
    train_size=int(X.shape[0]*train_ratio),
    val_size=int(X.shape[0]*val_ratio),
    X=X,
    y=y,
    M=None,
    shuffle=True,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.05,
  )
