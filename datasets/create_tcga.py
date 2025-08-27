"""TCGA cancer dataset. The prediction task is classification with 17 classes.
We try to predict the location of the tumor based on methylation data.
The 17 locations are:
  1. breast
  2. lung
  3. kidney
  4. brain
  5. ovary
  6. endometrial
  7. head and neck
  8. central nervous system
  9. thyroid
  10. prostate
  11. colon
  12. stomach
  13. bladder
  14. liver
  15. cervical
  16. bone marrow
  17. pancreas
We have also run feature selection to minimize the number of features to test on
as a preprocessing step. This was carried out with STG (https://arxiv.org/abs/1810.04247). 
The 21 chosen features are:
  1. c7orf51
  2. def6
  3. dnase1l3
  4. efs
  5. foxe1
  6. gpr81
  7. gria2
  8. gsdmc
  9. hoxa9
  10. kaag1
  11. klf5
  12. loc283392
  13. ltbr
  14. lyplal1
  15. pon3
  16. pou3f3
  17. serpinb1
  18. st6galnac1
  19. tmem106a
  20. znf583
  21. znf790
"""


import sys
import os
import os.path as osp

import numpy as np
import pandas as pd

import torch

from datasets.preprocessing_utils import preprocess_and_save_data


# Set seed.
SEED = 4150
np.random.seed(SEED)
torch.manual_seed(SEED)

path = osp.join("datasets", "data", "tcga")
os.makedirs(osp.join(path), exist_ok=True)

train_ratio = 0.8
val_ratio = 0.1


def print_tcga_download_info():
  print("Please download the TCGA data from https://www.cancer.gov/ccg/research/genome-sequencing/tcga.")
  print("Data can also be found at https://gdac.broadinstitute.org/")
  print("The clinical labels must be downloaded (Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz)")
  print("and placed in datasets/data/tcga/ called clinical.csv")
  print("The methylation data must be downloaded (Methylation_Preprocess.Level_3.2016012800.0.0.tar.gz)")
  print("and placed in datasets/data/tcga/ called methylation.csv")


if __name__ == "__main__":
  # Load in labels.
  try:
    print("Loading labels from clinical.csv.")
    df_clinical = pd.read_csv(osp.join(path, "clinical.csv"), low_memory=False)
    print("Labels loaded.")
  except FileNotFoundError:
    print("Failed to load labels from clinical.csv.")
    print_tcga_download_info()
    sys.exit()

  # Process labels.
  mapping_dict = {
    "breast": 0,
    "lung": 1,
    "kidney": 2,
    "brain": 3,
    "ovary": 4,
    "endometrial": 5,
    "head and neck": 6,
    "central nervous system": 7,
    "thyroid": 8,
    "prostate": 9,
    "colon": 10,
    "stomach": 11,
    "bladder": 12,
    "liver": 13,
    "cervical": 14,
    "bone marrow": 15,
    "pancreas": 16,
  }

  subject_column_name = "Unnamed: 0"
  subject_indices = pd.Index([])
  for site in mapping_dict.keys():
    subject_indices = subject_indices.union(df_clinical[ df_clinical["tumor_tissue_site"] == site ].index)
  subject_indices = list(df_clinical[subject_column_name].iloc[subject_indices])

  y = df_clinical.loc[df_clinical[subject_column_name].isin(subject_indices)]["tumor_tissue_site"].values
  y = np.array([mapping_dict[i] for i in y])


  # Load features.
  try:
    print("Loading features from methylation.csv.")
    df_methylation = pd.read_csv(osp.join(path, "methylation.csv"), low_memory=False)
    print("Features loaded.")
  except FileNotFoundError:
    print("Failed to load features from methylation.csv.")
    print_tcga_download_info()
    sys.exit()

  # Process features.
  # Find the samples that match the classes.
  X = df_methylation.loc[df_methylation[subject_column_name].isin(subject_indices)].values[:, 1:].astype(float)

  # Remove features with more than 15% missingness.
  feature_ids = []
  for i in range(X.shape[1]):
    if np.mean(np.isnan(X[:, i])) <= 0.15:
      feature_ids.append(i)
  feature_ids = np.array(feature_ids)
  X = X[:, feature_ids]

  # Remove samples from X and y that have more than 10% missing feature values.
  batch_ids = []
  for i in range(X.shape[0]):
    if np.mean(np.isnan(X[i])) <= 0.1:
      batch_ids.append(i)
  batch_ids = np.array(batch_ids)
  X = X[batch_ids]
  y = y[batch_ids]

  # 21 Best features chosen by STG.
  best_features = np.array([1826, 3324, 3518, 3751, 4523, 5068, 5104, 5143, 
                            5515, 6050, 6320, 6743, 6930, 6961, 9160, 9177, 
                            10608, 11528, 12125, 13570, 13643])

  X = X[:, best_features]
  M = 1.0 - np.isnan(X)
  X = np.where(np.isnan(X), 0, X)

  X = torch.tensor(X).float()
  M = torch.tensor(M).float()
  y = torch.tensor(y).long()


  dataset_dict = {
    "dataset": "tcga",
    "num_con_features": X.shape[1],
    "num_cat_features": 0,
    "most_categories": 0,
    "out_dim": int(y.max().item()+1),
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
    M=M,
    shuffle=True,
    num_bins=200,
    size_normal=1e-5,
    ratio_uniform=0.05,
  )
