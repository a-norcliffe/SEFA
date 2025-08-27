"""Two preprocessing shared modules:
1. Standardizer: Takes data and makes it mean 0 and std 1.
2. Vectorised Empirical CDF calculator, used to preprocess data specifically
for our model in accordance with https://arxiv.org/abs/1804.06216.
We also include a function to preprocess and save general data, this avoids
repeated code in the dataset scripts.
"""


import os.path as osp

import torch



class Standardizer:
  """Standardizes torch tensors to be mean 0 and std 1."""
  def __init__(self):
    pass

  @torch.no_grad()
  def fit(self, x, mask=None):
    if mask is None:
      self.mean = torch.mean(x, dim=0)
      self.std = torch.std(x, dim=0)
    else:
      self.mean = torch.sum(x*mask, dim=0) / torch.sum(mask, dim=0)
      # Bessel's correction.
      var = torch.sum(((x-self.mean)**2)*mask, dim=0)/(torch.sum(mask, dim=0)-1.0)
      self.std = var**0.5

  @torch.no_grad()
  def transform(self, x, mask=None):
    x = (x - self.mean) / self.std
    if mask is not None:
      x = x*mask
    return x


class VectorEmpiricalCDF:
  """A torch vectorised empirical cdf calculator.
  We calculate by finding the locations of the quantiles, and then we
  linearly interpolate a general point. Finally we apply a Standard Normal
  inverse CDF to get the processed values.
  args:
    num_bins: int, the number of bins to use for the empirical cdf.
    size_normal: float, the standard deviation of the normal noise to add to the
                 data. This is added to prevent the cdf from being a step if
                 continuous values only fall into a finite set.
    ratio_uniform: float, the ratio of uniform noise to add to the data. We add
                  points sampled uniformly in the range of the data to prevent,
                  the cdf from losing some of the spatial information.
  """
  def __init__(self, num_bins=200, size_normal=1e-5, ratio_uniform=0.1):
    self.num_bins = num_bins
    self.size_normal = size_normal
    self.ratio_uniform = ratio_uniform

  def linspace_batched(self, start, stop, steps):
    ints = (torch.arange(steps)).unsqueeze(0)
    dx = ((stop - start)/(steps - 1.0)).unsqueeze(-1)
    return start.unsqueeze(-1) + dx * ints

  @torch.no_grad()
  def fit(self, x, mask=None):
    # If we have missing values we sample from the real data to fill them in.
    # This is ok since the CDFs are used to find quantiles, and each
    # feature is treated independently.
    # NOTE this assumes there are at least some non-missing values.
    if mask is not None:
      for f in range(x.shape[-1]):
        m_tmp = mask[:, f]
        real_x = x[:, f][torch.where(m_tmp)[0]]
        sampled_real_ids = torch.multinomial(torch.ones(real_x.shape[0]), x.shape[0], replacement=True)
        sampled_real_x = real_x[sampled_real_ids]
        x[:, f] = m_tmp*x[:, f] + (1-m_tmp)*sampled_real_x

    # Add normal noise.
    x = x + self.size_normal*torch.randn_like(x)

    # Concatenate uniform data.
    min = torch.min(x, dim=0)[0]
    max = torch.max(x, dim=0)[0]
    uniform_data = (max-min)*torch.rand(size=(int(self.ratio_uniform*x.shape[0]), x.shape[-1])) + min
    x = torch.cat([x, uniform_data], dim=0)

    # Sort the data and find the quantiles.
    x = torch.sort(x, dim=0)[0]
    batchsize = x.shape[0]
    num_features = x.shape[1]

    # We start by looking at equally spaced quantiles. Equally spaced cdfs looks
    # in detail at regions where CDF changes quickly.
    # Step separates train set into that num_bins equally spaced bins.
    step = int(batchsize/self.num_bins)

    # These are the quantile values, i.e. the first has to be the minimum,
    # and the last has to be the maximum. Carried out in a batched way.
    self.s = torch.empty((num_features, self.num_bins))
    self.s[:, 0] = x[0, :]
    self.s[:, -1] = x[-1, :]

    # These are the cdf values at the quantile values, i.e. range from 0.0 to 1.0.
    self.cdf = torch.empty((num_features, self.num_bins))
    # This is an unbiased estimate of the min and max values drawn uniformly.
    # Rather than 0.0 and 1.0.
    self.cdf[:, 0] = 1/(batchsize+1)
    self.cdf[:, -1] = batchsize/(batchsize+1)

    for bin in range(1, self.num_bins - 1):
      self.s[:, bin] = x[step*bin, :]  # What x value is at this quantile.
      self.cdf[:, bin] = step*bin/batchsize  # What fraction of the data are we.

    # We then look at bins that are equally spaced in the x axis, this will
    # look in closer details at regions where the CDF changes slowly, which were
    # not previously captured.
    equally_spaced = self.linspace_batched(x[0, :], x[-1, :], self.num_bins)
    ids = torch.searchsorted((x.T).contiguous(), equally_spaced.contiguous(), right=True)
    cdf_x = ids/batchsize
    s_x = torch.gather(input=x.T, index=torch.clamp(ids, min=0, max=batchsize-1), dim=-1)
    s_x[:, 0] = x[0, :]
    s_x[:, -1] = x[-1, :]
    cdf_x[:, 0] = 1/(batchsize+1)
    cdf_x[:, -1] = batchsize/(batchsize+1)

    # Finally we merge the two, sort  and remove the first and last since they
    # are duplicates.
    self.s = torch.cat([self.s, s_x], dim=-1)
    self.cdf = torch.cat([self.cdf, cdf_x], dim=-1)
    self.s = torch.sort(self.s, dim=-1)[0][:, 1:-1]
    self.cdf = torch.sort(self.cdf, dim=-1)[0][:, 1:-1]

    # m is the constant in y = mx + c
    self.m = (self.cdf[:, 1:] - self.cdf[:, :-1]) / (self.s[:, 1:] - self.s[:, :-1])
    self.m = torch.cat([torch.zeros(self.m.shape[0], 1), self.m, torch.zeros(self.m.shape[0], 1)], dim=-1)

  @torch.no_grad()
  def empirical_cdf(self, x):
    ids = torch.searchsorted(self.s.contiguous(), (x.T).contiguous(), right=True)
    m_ = torch.gather(input=self.m, index=ids, dim=-1).T
    # Shift ids back to correctly select the cdf and x_ values.
    clamped_ids = torch.clamp(ids-1, max=self.cdf.shape[-1]-1, min=0)
    c_ = torch.gather(input=self.cdf, index=clamped_ids, dim=-1).T
    x_ = torch.gather(input=self.s, index=clamped_ids, dim=-1).T
    return m_ * (x - x_) + c_

  @torch.no_grad()
  def transform(self, x, mask=None):
    x = torch.clamp(self.empirical_cdf(x), min=1e-7, max=1.0-1e-7)
    x = 2**0.5 * torch.erfinv(2*x - 1)
    if mask is not None:
      x = x*mask
    return x


@torch.no_grad()
def preprocess_and_save_data(path, dataset_dict, train_size, val_size,
                             X, y, M=None, shuffle=False, num_bins=100,
                             size_normal=0.0, ratio_uniform=0.0):
  # Common function to preprocess and save all data to avoid repeated code.
  # Start with simple checks.
  assert X.shape[0] == y.shape[0], "X and y must have the same number of rows."
  assert X.shape[1] == dataset_dict["num_con_features"] + dataset_dict["num_cat_features"], "dataset dict is not consistent with X features."
  assert train_size + val_size < X.shape[0], "Train + val size must be less than the number of rows in X."

  if M is not None:
    assert M.shape == X.shape, "M must have the same shape as X."
    assert torch.all(torch.logical_or(M == 0.0, M == 1.0)), "M must be a binary mask."
    pass
  else:
    M = torch.ones_like(X)

  # Shuffle (if required) and split into train, val, test.
  shuffle_ids = torch.randperm(X.shape[0]) if shuffle else torch.arange(X.shape[0])

  train_ids = shuffle_ids[:train_size]
  val_ids = shuffle_ids[train_size:train_size+val_size]
  test_ids = shuffle_ids[train_size+val_size:]

  X_train = X[train_ids]
  X_val = X[val_ids]
  X_test = X[test_ids]

  y_train = y[train_ids]
  y_val = y[val_ids]
  y_test = y[test_ids]

  M_train = M[train_ids]
  M_val = M[val_ids]
  M_test = M[test_ids]

  # Get log(p(y)) and add to dataset information (for ACFlow).
  log_py = get_y_log_probs_from_counts(y_train, dataset_dict["out_dim"])
  dataset_dict["log_class_probs"] = log_py

  # Save dataset information
  torch.save(dataset_dict, osp.join(path, "dataset_dict.pt"))

  # Save the y and M tensors before processing features.
  torch.save(y_train.long(), osp.join(path, "y_train.pt"))
  torch.save(y_val.long(), osp.join(path, "y_val.pt"))
  torch.save(y_test.long(), osp.join(path, "y_test.pt"))

  torch.save(M_train.float(), osp.join(path, "M_train.pt"))
  torch.save(M_val.float(), osp.join(path, "M_val.pt"))
  torch.save(M_test.float(), osp.join(path, "M_test.pt"))

  # Process continuous features.
  num_con_features = dataset_dict["num_con_features"]
  if num_con_features != 0:
    X_con_train = X_train[:, :num_con_features]
    X_con_val = X_val[:, :num_con_features]
    X_con_test = X_test[:, :num_con_features]

    M_con_train = M_train[:, :num_con_features]
    M_con_val = M_val[:, :num_con_features]
    M_con_test = M_test[:, :num_con_features]

    M_con_train = None if torch.all(M_con_train) else M_con_train
    M_con_val = None if torch.all(M_con_val) else M_con_val
    M_con_test = None if torch.all(M_con_test) else M_con_test

    X_con_train_cdf, X_con_val_cdf, X_con_test_cdf = process_continuous_features_cdf(
      X_con_train, X_con_val, X_con_test, M_con_train, M_con_val, M_con_test,
      num_bins, size_normal, ratio_uniform
    )
    X_con_train_std, X_con_val_std, X_con_test_std = process_continuous_features_std(
      X_con_train, X_con_val, X_con_test, M_con_train, M_con_val, M_con_test,
    )

  else:
    X_con_train_cdf = torch.empty(size=(X_train.shape[0], 0))
    X_con_val_cdf = torch.empty(size=(X_val.shape[0], 0))
    X_con_test_cdf = torch.empty(size=(X_test.shape[0], 0))

    X_con_train_std = torch.empty(size=(X_train.shape[0], 0))
    X_con_val_std = torch.empty(size=(X_val.shape[0], 0))
    X_con_test_std = torch.empty(size=(X_test.shape[0], 0))

  # Process categorical features.
  num_cat_features = dataset_dict["num_cat_features"]
  if num_cat_features != 0:
    X_cat_train = X_train[:, -num_cat_features:]
    X_cat_val = X_val[:, -num_cat_features:]
    X_cat_test = X_test[:, -num_cat_features:]
  else:
    X_cat_train = torch.empty(size=(X_train.shape[0], 0))
    X_cat_val = torch.empty(size=(X_val.shape[0], 0))
    X_cat_test = torch.empty(size=(X_test.shape[0], 0))


  X_train_cdf = torch.cat((X_con_train_cdf.float(), X_cat_train.float()), dim=1)
  X_val_cdf = torch.cat((X_con_val_cdf.float(), X_cat_val.float()), dim=1)
  X_test_cdf = torch.cat((X_con_test_cdf.float(), X_cat_test.float()), dim=1)

  X_train_std = torch.cat((X_con_train_std.float(), X_cat_train.float()), dim=1)
  X_val_std = torch.cat((X_con_val_std.float(), X_cat_val.float()), dim=1)
  X_test_std = torch.cat((X_con_test_std.float(), X_cat_test.float()), dim=1)

  torch.save(X_train_cdf, osp.join(path, "X_train_cdf.pt"))
  torch.save(X_val_cdf, osp.join(path, "X_val_cdf.pt"))
  torch.save(X_test_cdf, osp.join(path, "X_test_cdf.pt"))

  torch.save(X_train_std, osp.join(path, "X_train_std.pt"))
  torch.save(X_val_std, osp.join(path, "X_val_std.pt"))
  torch.save(X_test_std, osp.join(path, "X_test_std.pt"))


def process_continuous_features_cdf(
  X_con_train, X_con_val, X_con_test, M_con_train, M_con_val, M_con_test,
  num_bins, size_normal, ratio_uniform,
):
  # CDF normalize the continuous features.
  empirical_cdf = VectorEmpiricalCDF(
    num_bins=num_bins,
    size_normal=size_normal,
    ratio_uniform=ratio_uniform,
  )
  empirical_cdf.fit(X_con_train, M_con_train)
  X_con_train_cdf = empirical_cdf.transform(X_con_train, M_con_train)
  X_con_val_cdf = empirical_cdf.transform(X_con_val, M_con_val)
  X_con_test_cdf = empirical_cdf.transform(X_con_test, M_con_test)

  return X_con_train_cdf, X_con_val_cdf, X_con_test_cdf


def process_continuous_features_std(
  X_con_train, X_con_val, X_con_test, M_con_train, M_con_val, M_con_test,
):
  # Make mean 0 and standard deviation 0.
  standardizer = Standardizer()
  standardizer.fit(X_con_train, M_con_train)
  X_con_train_std = standardizer.transform(X_con_train, M_con_train)
  X_con_val_std = standardizer.transform(X_con_val, M_con_val)
  X_con_test_std = standardizer.transform(X_con_test, M_con_test)

  return X_con_train_std, X_con_val_std, X_con_test_std


def get_y_log_probs_from_counts(y_train, num_classes):
  class_counts = torch.bincount(y_train, minlength=num_classes)
  y_log_probs = torch.log(class_counts.float() / class_counts.sum() + 1e-8)
  return y_log_probs