# stat535_project
Final project repo for STAT 535: Statistical Machine Learning at the University of Washington.

A detailed description of the problem statement and methods used can be found in the write-up: `535_project_report.pdf`. Preprocessing is done in `preprocessing.ipynb`. The model is a neural network implemented in `PyTorch` and can be found in `model.py`. All hyperparameters are specified in `hyperparameters.py`. Training occus in `train.py`, and `train.sbatch` is a Slurm script that runs the training loop on a CPU cluster. Finally, test set preprocessing and evaluation are done in `text.ipynb`.
