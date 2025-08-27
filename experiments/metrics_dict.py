"""Torch specific metrics that act on tensors.

NOTE: Classification metrics assume that y_pred is a probability distribution
over classes.
They use the library torchmetrics which splits into multiclass and binary
metrics. These metrics have conditions that account for binary or multiclass.
Classification metrics are accuracy, auroc.
"""

from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.functional.classification import binary_auroc

from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.functional.classification import multiclass_auroc


def torch_accuracy(y_pred, y):
  if y_pred.ndim == 2:
    if y_pred.shape[1] > 2:
      return multiclass_accuracy(y_pred, y, num_classes=y_pred.shape[1], average="micro").item()
    y_pred = y_pred[:, -1]
  return binary_accuracy(y_pred, y).item()


def torch_auroc(y_pred, y):
  if y_pred.ndim == 2:
    if y_pred.shape[1] > 2:
      return multiclass_auroc(y_pred, y, num_classes=y_pred.shape[1], average="macro").item()
    y_pred = y_pred[:, -1]
  return binary_auroc(y_pred, y).item()


# Dictionary where we can choose the metrics by name.
metrics_dict = {
  "accuracy": torch_accuracy,
  "auroc": torch_auroc,
}