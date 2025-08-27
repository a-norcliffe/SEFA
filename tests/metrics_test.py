"""Test torchmetrics with the sklearn equivalent metrics."""


import unittest

import numpy as np
import torch
import torch.nn as nn

from experiments.metrics_dict import torch_accuracy, torch_auroc
from sklearn.metrics import accuracy_score, roc_auc_score


batchsize = 2000
num_features = 5
num_hidden = 100
num_multiclass = 5
places = 5  # Number of decimal place to compare outputs.


def create_data(num_classes, batchsize=batchsize, num_features=num_features):
  x = torch.randn((batchsize, num_features))
  y = torch.randint(0, num_classes, (batchsize,)).long()
  model = nn.Sequential(
    nn.Linear(num_features, num_hidden),
    nn.ReLU(),
    nn.Linear(num_hidden, num_classes),
    nn.Softmax(dim=-1)
  )
  model.eval()
  y_preds = model(x)
  y_preds_np = y_preds.detach().cpu().numpy()
  y_np = y.detach().cpu().numpy()
  return y_preds, y, y_preds_np, y_np


class TestAccuracy(unittest.TestCase):

  def test_multiclass_accuracy(self):
    y_pred, y, y_pred_np, y_np = create_data(num_multiclass)
    self.assertAlmostEqual(torch_accuracy(y_pred, y), accuracy_score(y_np, np.argmax(y_pred_np, axis=-1)), places=places)

  def test_binary_accuracy(self):
    y_pred, y, y_pred_np, y_np = create_data(2)
    self.assertAlmostEqual(torch_accuracy(y_pred, y), accuracy_score(y_np, np.argmax(y_pred_np, axis=-1)), places=places)


class TestAUROC(unittest.TestCase):

  def test_multiclass_auroc(self):
    y_pred, y, y_pred_np, y_np = create_data(num_multiclass)
    self.assertAlmostEqual(torch_auroc(y_pred, y), roc_auc_score(y_np, y_pred_np, multi_class="ovr"), places=places)

  def test_binary_auroc(self):
    y_pred, y, y_pred_np, y_np = create_data(2)
    self.assertAlmostEqual(torch_auroc(y_pred, y), roc_auc_score(y_np, y_pred_np[:, 1]), places=places)



if __name__ == "__main__":
  unittest.main()