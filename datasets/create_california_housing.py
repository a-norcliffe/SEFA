"""The California Housing dataset https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
This predicts the median house value for houses in a given district.
The features are:
  1. MedInc median income in block group
  2. HouseAge median house age in block group
  3. AveRooms average number of rooms per household
  4. AveBedrms average number of bedrooms per household
  5. Population block group population
  6. AveOccup average number of household members
  7. Latitude block group latitude
  8. Longitude block group longitude
We convert this into a classification task by converting the median house price
into four equally sized bins, to lead to four equally sized classes.
"""



import os
import os.path as osp

import numpy as np
import torch

from sklearn.datasets import fetch_california_housing
from datasets.preprocessing_utils import preprocess_and_save_data


# Set seed.
SEED = 2538
np.random.seed(SEED)
torch.manual_seed(SEED)

path = osp.join("datasets", "data", "california_housing")
os.makedirs(path, exist_ok=True)

train_ratio = 0.8
val_ratio = 0.1
num_classes = 4



if __name__ == "__main__":
  X, y = fetch_california_housing(return_X_y=True, as_frame=True)
  X = X.values
  y = y.values

  y_sorted = np.sort(y)
  bins = []
  for i in range(1, num_classes):
    bins.append(y_sorted[int(i*len(y_sorted)/num_classes)])
  bins.append(y_sorted[-1])

  y = np.digitize(y, bins, right=True)
  assert np.all(np.unique(y) == np.arange(num_classes)), f"Labels should be consecutive integers from 0 to {num_classes - 1}"

  X = torch.tensor(X).float()
  y = torch.tensor(y).long()


  dataset_dict = {
    "dataset": "california_housing",
    "num_con_features": X.shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": num_classes,
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