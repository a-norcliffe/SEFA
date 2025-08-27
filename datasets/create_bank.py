"""Bank Marketing data.
Downloaded from https://archive.ics.uci.edu/dataset/222/bank+marketing.
The task is to predict whether the consumer bought the product that the bank
was marketing. The features contain information like age, job, savings etc.
"""


import sys

import os
import os.path as osp

import numpy as np
import pandas as pd

import torch

from datasets.preprocessing_utils import preprocess_and_save_data


# Set seed.
SEED = 9675
np.random.seed(SEED)
torch.manual_seed(SEED)

path = osp.join("datasets", "data", "bank")
os.makedirs(path, exist_ok=True)

train_ratio = 0.8
val_ratio = 0.1


if __name__ == "__main__":
  try:
    print("Loading data.")
    df = pd.read_csv(osp.join(path, "bank-full.csv"), header=0, delimiter=";")
    print("Data loaded.")
  except FileNotFoundError:
    print("Please download the Bank Marketing data from https://archive.ics.uci.edu/dataset/222/bank+marketing")
    print("and place it in the datasets/data/bank folder.")
    print("Or use the Linux commands (worked on 08/08/2025):")
    print("> wget -P datasets/data/bank https://archive.ics.uci.edu/static/public/222/bank+marketing.zip")
    print("> unzip datasets/data/bank/bank+marketing -d datasets/data/bank")
    print("> unzip datasets/data/bank/bank -d datasets/data/bank")
    print("> rm datasets/data/bank/bank-additional.zip")
    print("> rm datasets/data/bank/bank.zip")
    print("> rm datasets/data/bank/bank+marketing.zip")
    print("> rm datasets/data/bank/bank-names.txt")
    print("> rm datasets/data/bank/bank.csv")
    sys.exit()

  y = df["y"].apply(lambda x: 1 if x == "yes" else 0).values


  # Continuous features.
  age = df["age"].values
  balance = df["balance"].values
  duration = df["duration"].values
  campaign = df["campaign"].values
  previous = df["previous"].values
  month_mapping = {
    "jan": 0,
    "feb": 31,
    "mar": 60,
    "apr": 91,
    "may": 121,
    "jun": 152,
    "jul": 182,
    "aug": 213,
    "sep": 244,
    "oct": 274,
    "nov": 305,
    "dec": 335,
  }
  month = df["month"].values
  day = df["day"].values
  date = np.array([month_mapping[m] + d for m, d in zip(month, day)])
  pdays = df["pdays"].values
  pdays = np.where(pdays==-1, -500, pdays)  # Shift to -500 so there is a clear distinction, after standardising there is less skew.

  X_con = np.stack([age, balance, duration, campaign, previous, date, pdays], axis=-1)


  # Categorical features.
  job = df["job"].values
  mapping = {
    "management": 0,
    "technician": 1,
    "entrepreneur": 2,
    "blue-collar": 3,
    "retired": 4,
    "admin.": 5,
    "services": 6,
    "self-employed": 7,
    "unemployed": 8,
    "housemaid": 9,
    "student": 10,
    "unknown": np.nan,
  }
  job = np.array([mapping[j] for j in job])

  marital = df["marital"].values
  mapping = {
    "single": 0,
    "married": 1,
    "divorced": 2,
  }
  marital = np.array([mapping[m] for m in marital])

  education = df["education"].values
  mapping = {
    "primary": 0,
    "secondary": 1,
    "tertiary": 2,
    "unknown": np.nan
  }
  education = np.array([mapping[e] for e in education])

  default = df["default"].values
  mapping = {
    "no": 0,
    "yes": 1,
  }
  default = np.array([mapping[d] for d in default])

  housing = df["housing"].values
  mapping = {
    "no": 0,
    "yes": 1,
  }
  housing = np.array([mapping[h] for h in housing])

  loan = df["loan"].values
  mapping = {
    "no": 0,
    "yes": 1,
  }
  loan = np.array([mapping[l] for l in loan])

  contact = df["contact"].values
  mapping = {
    "cellular": 0,
    "telephone": 1,
    "unknown": np.nan
  }
  contact = np.array([mapping[c] for c in contact])

  poutcome = df["poutcome"].values # Outcome of previous marketing campaign.
  mapping = {
    "failure": 0,
    "success": 1,
    "unknown": 2,
    "other": np.nan
  }
  poutcome = np.array([mapping[p] for p in poutcome])

  X_cat = np.stack([job, marital, education, default, housing, loan, contact, poutcome], axis=-1)

  X = np.concatenate([X_con, X_cat], axis=-1).astype(float)
  M = 1.0 - np.isnan(X)
  X = np.where(np.isnan(X), 0, X)

  X = torch.tensor(X).float()
  M = torch.tensor(M).float()
  y = torch.tensor(y).long()

  dataset_dict = {
    "dataset": "bank",
    "num_con_features": X_con.shape[-1],
    "num_cat_features": X_cat.shape[-1],
    "most_categories": int(np.nanmax(X_cat))+1,
    "out_dim": int(y.max().item()+1),
    "metric": "auroc",
    "max_dim": None,
  }

  preprocess_and_save_data(
    path=path,
    dataset_dict=dataset_dict,
    train_size=int(X.shape[0]*train_ratio),
    val_size=int(X.shape[0]*val_ratio),
    X=X,
    y=y,
    M=M,
    shuffle=True,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.05,
  )