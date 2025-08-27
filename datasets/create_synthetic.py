"""Data is created from the synthetic INVASE experiments:
https://openreview.net/forum?id=BJg_roAcK7
This is useful for our task since the feature order is definitely
instance-wise in this case, whereas real world datasets can have a population
level optimal order.

There is a small discrepancy between the paper and the code. The code is here:
https://github.com/jsyoon0823/INVASE/blob/master/data_generation.py
We use the code's version of the dataset.
Paper reports logit 3 as:
L3 = -10sin(2x_7) + 2|x_8| + x_9 + exp(-x_10)
The code and what we use reports logit 3 as:
L3 = -10sin(0.2x_7) + |x_8| + x_9 + exp(-x_10) - 2.4

Note: we have also made changes to logit 1 and 2 to improve all models'
performances. Logit 1 was:
L1 = x_1*x_2
we change it to:
L1 = 4*x_1*x_2
Logit 2 was:
L2 = x_3^2 + x_4^2 + x_5^2 + x_6^2 - 4.0
we have changed it to
L2 = 1.2*(x_3^2 + x_4^2 + x_5^2 + x_6^2) - 4.2

This is because this creates a logit that is used to sample, and the original
use of these datasets was for feature selection, and not looking at performance.
"""

import os
import os.path as osp

import numpy as np
import torch

from datasets.preprocessing_utils import preprocess_and_save_data


# Set seed.
SEED = 8139
np.random.seed(SEED)
torch.manual_seed(SEED)

train_size = 60_000
val_size = 10_000
test_size = 10_000


def create_data(batchsize, num_features=11, rule=1):
  x = np.random.normal(size=(batchsize, num_features))

  syn1 = 4*x[:, 0]*x[:, 1]
  syn2 = 1.2*np.sum(x[:, 2:6]**2, axis=-1) - 4.2
  syn3 = -10*np.sin(0.2*x[:, 6]) + abs(x[:, 7]) + x[:, 8] + np.exp(-x[:, 9]) - 2.4

  if rule == 1:
    logit1 = syn1
    logit2 = syn2
  elif rule == 2:
    logit1 = syn1
    logit2 = syn3
  elif rule == 3:
    logit1 = syn2
    logit2 = syn3
  else:
    raise ValueError(f"Unknown rule {rule}")

  feature_11_less_0 = (x[:, -1] < 0).astype(np.float32)
  logit = feature_11_less_0*logit1 + (1.0-feature_11_less_0)*logit2

  y = np.random.binomial(1, 1/(1+np.exp(logit)), size=batchsize)
  return x, y


if __name__ == "__main__":
  for rule in [1, 2, 3]:
    # Create data.
    X, y = create_data(train_size+val_size+test_size, rule=rule)
    X = torch.tensor(X).float()
    y = torch.tensor(y).long()

    # Make folder.
    path = osp.join("datasets", "data", f"syn{rule}")
    os.makedirs(path, exist_ok=True)

    dataset_dict = {
      "dataset": f"syn{rule}",
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
      train_size=train_size,
      val_size=val_size,
      X=X,
      y=y,
      M=None,
      shuffle=False,
      num_bins=200,
      size_normal=0.0,
      ratio_uniform=0.0,
    )
