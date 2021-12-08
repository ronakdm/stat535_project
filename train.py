import torch.nn as nn
import pickle
import numpy as np
import sys

from utils import seed_everything, set_device, get_train_val_dataloaders, train
from model import FeedForwardRegressionNet
from hyperparameters import sample_hyperparameters

from torch.utils.data import TensorDataset
from torch.optim import Adam


# Hyperparmaters.

# When using with Slurm.
job_id = int(sys.argv[1])
hyperparameters = sample_hyperparameters()

HIDDEN_DIMS = hyperparameters["hidden_dims"]
LEARNING_RATE = hyperparameters["learning_rate"]
EMBED_DIM0 = hyperparameters["embed_dim0"]
EMBED_DIM1 = hyperparameters["embed_dim1"]
EPOCHS = hyperparameters["epochs"]
DROPOUT = hyperparameters["dropout"]
# WEIGHT_DECAY = hyperparameters["weight_decay"]


batch_size = 64
val_size = 0.1

device = set_device()

X = pickle.load(open("preprocessed_data/X_train_tensor.pkl", "rb"))
y = pickle.load(open("preprocessed_data/y_train_tensor.pkl", "rb"))

dataset = TensorDataset(X, y)

train_dataloader, validation_dataloader = get_train_val_dataloaders(
    dataset, val_size, batch_size
)

input_dim = X.shape[1] - 2
vocab_size0 = len(np.unique(X[:, 0]))
vocab_size1 = len(np.unique(X[:, 1]))
output_dim = 1

model = FeedForwardRegressionNet(
    input_dim,
    vocab_size0,
    vocab_size1,
    EMBED_DIM0,
    EMBED_DIM1,
    HIDDEN_DIMS,
    output_dim,
    DROPOUT,
)

# optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.MSELoss()

seed_everything(42)

training_stats = train(
    model,
    EPOCHS,
    train_dataloader,
    validation_dataloader,
    criterion,
    optimizer,
    device=device,
    SSY=y.var() * len(y),
)

training_stats["hidden_dims"] = HIDDEN_DIMS
training_stats["learning_rate"] = LEARNING_RATE
training_stats["embed_dim0"] = EMBED_DIM0
training_stats["embed_dim1"] = EMBED_DIM1
training_stats["epochs"] = EPOCHS

training_stats["job_id"] = job_id
pickle.dump(training_stats, open(f"results/training_stats_{job_id}", "wb"))

