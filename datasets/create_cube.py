"""Creates data from established cube dataset:
https://realworld-sdm.github.io/paper/21.pdf
https://link.springer.com/article/10.1007/s13042-012-0092-x
https://papers.nips.cc/paper_files/paper/2018/hash/e5841df2166dd424a57127423d276bbe-Abstract.html

20 features, 8 classes. Features 11-20 are noise. And for each class only some
features are relevant:
1: [f1, f2, f3] ~ N([0, 0, 0], [0.1, 0.1, 0.1])
2: [f2, f3, f4] ~ N([1, 0, 0], [0.1, 0.1, 0.1])
3: [f3, f4, f5] ~ N([0, 1, 0], [0.1, 0.1, 0.1])
4: [f4, f5, f6] ~ N([1, 1, 0], [0.1, 0.1, 0.1])
5: [f5, f6, f7] ~ N([0, 0, 1], [0.1, 0.1, 0.1])
6: [f6, f7, f8] ~ N([1, 0, 1], [0.1, 0.1, 0.1])
7: [f7, f8, f9] ~ N([0, 1, 1], [0.1, 0.1, 0.1])
8: [f8, f9, f10] ~ N([1, 1, 1], [0.1, 0.1, 0.1])

All other features including 11-20 are noise sampled from N(0.5, 0.3). Note that
here N(m, s) is the normal distribution with mean m and STANDARD DEVIATION s
NOT VARIANCE s.
"""


import os
import os.path as osp

import numpy as np

import torch

from datasets.preprocessing_utils import preprocess_and_save_data


# Set seed.
SEED = 8917
np.random.seed(SEED)
torch.manual_seed(SEED)

path = osp.join("datasets", "data", "cube")
os.makedirs(path, exist_ok=True)

train_size = 60_000
val_size = 10_000
test_size = 10_000
noise_mu = 0.5
noise_sig = 0.3
num_features = 20
num_classes = 8


dataset_dict = {
  "dataset": "cube",
  "num_con_features": num_features,
  "num_cat_features": 0,
  "most_categories": 0,
  "out_dim": num_classes,
  "metric": "accuracy",
  "max_dim": None,
}


# Create the means and sigmas for the cube data.
means = torch.full((num_classes, num_features), noise_mu)
means[0, 0:3] = torch.tensor([0.0, 0.0, 0.0])
means[1, 1:4] = torch.tensor([1.0, 0.0, 0.0])
means[2, 2:5] = torch.tensor([0.0, 1.0, 0.0])
means[3, 3:6] = torch.tensor([1.0, 1.0, 0.0])
means[4, 4:7] = torch.tensor([0.0, 0.0, 1.0])
means[5, 5:8] = torch.tensor([1.0, 0.0, 1.0])
means[6, 6:9] = torch.tensor([0.0, 1.0, 1.0])
means[7, 7:10] = torch.tensor([1.0, 1.0, 1.0])

sigs = torch.full((num_classes, num_features), noise_sig)
sigs[0, 0:3] = torch.tensor([0.1, 0.1, 0.1])
sigs[1, 1:4] = torch.tensor([0.1, 0.1, 0.1])
sigs[2, 2:5] = torch.tensor([0.1, 0.1, 0.1])
sigs[3, 3:6] = torch.tensor([0.1, 0.1, 0.1])
sigs[4, 4:7] = torch.tensor([0.1, 0.1, 0.1])
sigs[5, 5:8] = torch.tensor([0.1, 0.1, 0.1])
sigs[6, 6:9] = torch.tensor([0.1, 0.1, 0.1])
sigs[7, 7:10] = torch.tensor([0.1, 0.1, 0.1])


def create_data(size_per_class):
  """Creates data of given num_classes * size. i.e. size of each class."""
  X_data = torch.distributions.normal.Normal(means, sigs).sample([size_per_class])
  X_data = X_data.view(-1, num_features).float()
  y_data = torch.arange(num_classes).repeat(size_per_class).long()
  return X_data, y_data


if __name__ == "__main__":
  # Generate the cube data.
  X, y = create_data(int((train_size+val_size+test_size)/num_classes))
  assert X.shape[0] == train_size + val_size + test_size, "create_data has not worked as expected for size creation"
  assert torch.all(y[0:8] == torch.arange(num_classes)), "create_data has not worked as expected for y creation"

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
