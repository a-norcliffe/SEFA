"""Model constants used throughout."""


import numpy as np


log_eps = 1e-8
min_sig = 1e-3
half_log_2pi = 0.5*np.log(2*np.pi)

# Max batchsize for acquisition scoring, since it can be memory intensive.
# This can be changed if more or less memory is available.
acquisition_batch_limit = 10_000_000

# These are for LR Reduce on Plateau Schedulers.
cooldown = 0
lr_factor = 0.2
min_lr = 1e-7