# Stochastic Encodings for Active Feature Acquisition

[![Paper](https://img.shields.io/badge/Paper-ICML%202025-red.svg)](https://openreview.net/forum?id=MjVmVakGdx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/a-norcliffe/SEFA/blob/master/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

Public code for the ICML 2025 paper [**Stochastic Encodings for Active Feature Acquisition**](https://arxiv.org/abs/2508.01957)

(
[Alexander Norcliffe](https://scholar.google.com/citations?user=BbeDr6EAAAAJ&hl=en),
[Changhee Lee](https://sites.google.com/view/actionable-intelligence/professor?authuser=0),
[Fergus Imrie](https://fimrie.github.io/),
[Mihaela van der Schaar](https://www.vanderschaar-lab.com/prof-mihaela-van-der-schaar/),
[Pietro Liò](https://www.cl.cam.ac.uk/~pl219/)
)

In the real world not all information to solve a problem is all available at once. An effective machine learning model should be able to look at its current information and decide what to measure next to improve its prediction. This problem is known as Active Feature Acquisition. Doctors do this when they diagnose a patient: based on their existing observations they choose what test to conduct next. Existing solutions tend to use Reinforcement Learning, which can suffer from training instability, or maximize the conditional mutual information of the label and unobserved features, which can lead to myopic acquisitions. We propose a novel latent variable approach to Active Feature Acquisition that avoids both issues.

![SEFA Prediction](https://github.com/a-norcliffe/SEFA/raw/main/readme_figures//sefa_prediction.svg)
*The architecture of SEFA. Each feature is stochastically encoded, samples can be pushed through a predictor network and averaged to get an overall prediction.*

![SEFA Acquisition](https://github.com/a-norcliffe/SEFA/raw/main/readme_figures//sefa_acquisition.svg)
*The acquisition process in SEFA. SEFA takes many latent samples, and uses gradient information of each sample's prediction to score features. This is averaged across samples and weighted based on the existing prediction to place more focus on more likely classes.*


## Abstract

Active Feature Acquisition is an instance-wise, sequential decision making problem. The aim is to dynamically select which feature to measure based on current observations, independently for each test instance. Common approaches either use Reinforcement Learning, which experiences training difficulties, or greedily maximize the conditional mutual information of the label and unobserved features, which makes myopic acquisitions. To address these shortcomings, we introduce a latent variable model, trained in a supervised manner. Acquisitions are made by reasoning about the features across many possible unobserved realizations in a stochastic latent space. Extensive evaluation on a large range of synthetic and real datasets demonstrates that our approach reliably outperforms a diverse set of baselines.



## Installation and Usage
We used `python 3.8` for this project. To setup the virtual environment and necessary packages, please run the following commands:

```
$ conda create --name sefa_env python=3.8
$ conda activate sefa_env
```

Apart from standard libraries, we also used:

- matplotlib 3.7.2
- numpy 1.24.3
- scikit-learn 1.3.0
- pandas 1.5.3
- pytorch 1.13.1
- torchmetrics 1.1.2
- torchvision 0.14.1
- tabulate 0.9.0


### Running SEFA on your Data
We have included a notebook, `example.ipynb`, which demonstrates SEFA on Syn1 and two additional synthetic tasks. It shows how to set up the data for SEFA, the training and the inference.


### Tests
Tests are in the `tests` folder and can be run from the command line (from the home directory), for example:

```
$ python -m tests.acflow_test
$ python -m tests.acquisition_test
```

Alternatively, all tests can be run with one command using a bash script:

```
$ bash tests/run_tests.sh
```


### Datasets
Code for creating the datasets can be found in the `datasets` folder, and run using the command line (from the home directory), for example:

```
$ python -m datasets.create_synthetic
$ python -m datasets.create_cube
```

Alternatively, all datasets can be created with one command using a bash script:

```
$ bash datasets/create_datasets.sh
```

Note, the raw data must be downloaded from the respective sources. Details can be found in each dataset's creation script.


### Main Experiments
Experiments can be run from the command line (from the home direcetory). Examples of hyperparameter sweeps are:

```
$ python -m experiments.run_sweeps --dataset syn1 --model sefa --first_config 1 --last_config 4
$ python -m experiments.run_sweeps --dataset bank --model dime --first_config 3 -- last_config 6
$ python -m experiments.run_sweeps --dataset metabric --model eddi --first_config 9 --last_config 9
```

After running all 9 sweeps on a given model on a given dataset, main training can be run from the command line. Examples are:

```
$ python -m experiments.run_main_training --dataset california_housing --model gdfs --first_run 1 --last_run 5
$ python -m experiments.run_main_training --dataset miniboone --model opportunistic --first_run 3 --last_run 4
$ python -m experiments.run_main_training --dataset tcga --model fixed_mlp --first_run 1 --last_run 1
```

After training all models on all datasets 5 times, inference can be run from the command line. Examples are:

```
$ python -m experiments.run_main_inference --dataset mnist
$ python -m experiments.run_main_inference --dataset fashion_mnist
```


### Ablations
Ablations can be run from the command line (from the home directory). Examples are:

```
python -m experiments.run_ablations --dataset syn2 --ablation beta
python -m experiments.run_ablations --dataset syn3 --ablation train_sample
```

Only a few ablations require training a new model. After these are trained, the inference can be run from the command line. Examples are:

```
$ python -m experiments.run_ablations_inference --dataset tcga
$ python -m experiments.run_ablations_inference --dataset metabric
```

### Tables and Figures
After running all the inference. The `evaluation.ipynb` notebook provides code to create the tables and figures in the paper.


## Notes

- The SEFA code is in `models/sefa.py`, if you wish to modify it directly. It uses
a base class from `models/base.py`. It also uses some shared functions and constants
from the `models` folder. Importantly, SEFA can be quite memory intensive if
many acquisition/train samples are used or if the batchsize is large.
This can be controlled by reducing the `acquisition_batch_limit` constant in `models/constants.py`.
- If SEFA uses continuous and categorical features, all continuous must be first, then all categorical.
- GSMRL can only be tuned/trained when there are *trained* fixed MLPs and ACFlow models.
- After sweeping hyperparameters, there is code that finds the best performing
hyperparameter set automatically when training. To override this, you can save a "fake" result so that the training code finds this as the best.
As an example, for cube, dime config 2, run:

`torch.save({"mean": 500, "std": 0.1}, "experiments/hyperparameters/tuning_results/cube/dime/config_2.pt")`

- The code is setup for 5 repeats, this can be changed in the inference code and
evaluation notebook.
- The sensitivity analysis requires certain ablations to be run, see the sensitivity inference to see these.
- All commands should be run from the home directory.




## Citation
If our paper or code helped you in your own research, please cite our work as:

```
@inproceedings{norcliffe2025sefa,
  title={{Stochastic Encodings for Active Feature Acquisition}},
  author={Norcliffe, Alexander and Lee, Changhee and Imrie, Fergus and van der Schaar, Mihaela and Li\`{o}, Pietro},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=MjVmVakGdx}
}
```

## Acknowledgements
The results in the TCGA experiment are based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga. We thank the ICML reviewers for their comments and suggestions. We also thank Lucie Charlotte Magister and Iulia Duta for providing feedback. Alexander Norcliffe is supported by a GSK plc grant. Changhee Lee is supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00358602) and the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT), Artificial Intelligence Graduate School Program (No. RS-2019-II190079, Korea University) and the Artificial Intelligence Star Fellowship Support Program to nurture the best talents (No. RS-2025-02304828).