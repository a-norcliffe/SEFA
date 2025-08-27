"""Fashion MNIST dataset, we take the Fashion MNIST dataset
https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
and we convert this to be tabular by using STG to select the 20 most
predictive features.
"""


import os
import os.path as osp

import numpy as np

import torch
import torchvision.transforms as T
from torchvision.datasets import FashionMNIST

from datasets.preprocessing_utils import preprocess_and_save_data


# Set seed.
SEED = 7508
np.random.seed(SEED)
torch.manual_seed(SEED)

path = osp.join("datasets", "data", "fashion_mnist")
os.makedirs(path, exist_ok=True)

train_size = 50_000
val_size = 10_000


if __name__ == "__main__":
  fmnist_train_val = FashionMNIST(root=path, train=True, download=True, transform=T.ToTensor())
  fmnist_test = FashionMNIST(root=path, train=False, download=True, transform=T.ToTensor())

  X_train_val = torch.stack([batch[0].view(-1) for batch in fmnist_train_val], dim=0).float()
  y_train_val = torch.tensor([batch[1] for batch in fmnist_train_val]).long()

  X_test = torch.stack([batch[0].view(-1) for batch in fmnist_test], dim=0).float()
  y_test = torch.tensor([batch[1] for batch in fmnist_test]).long()

  # Best features chosen by STG.
  best_features = np.array([10, 38, 121, 146, 202, 246, 248, 341, 343, 362, 406,
                            434, 454, 490, 546, 574, 580, 602, 742, 770])

  X_train_val = X_train_val[:, best_features]
  X_test = X_test[:, best_features]

  # Shuffle train set to get train and val set.
  shuffle_ids = np.random.permutation(X_train_val.shape[0])
  X_train_val = X_train_val[shuffle_ids]
  y_train_val = y_train_val[shuffle_ids]

  X = torch.cat([X_train_val, X_test], dim=0).float()
  y = torch.cat([y_train_val, y_test], dim=0).long()

  dataset_dict = {
    "dataset": "fashion_mnist",
    "num_con_features": X.shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": 10,
    "metric": "accuracy",
    "max_dim": None,
  }

  preprocess_and_save_data(
    path=path,
    dataset_dict=dataset_dict,
    train_size=train_size,
    val_size=val_size,
    X=X,
    y=y,
    M=None,
    shuffle=False,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.2,
  )