"""Same as models_cpu_test, but we test on a GPU, to check
all devices are as expected.
"""

import sys

import unittest

import torch

from tests.models_cpu_test import TestModels




if torch.cuda.is_available():
  device = torch.device(f"cuda:0")
else:
  print("No GPU to test on, exiting this test.")
  sys.exit()


if __name__ == "__main__":
  print(device)
  unittest.main()