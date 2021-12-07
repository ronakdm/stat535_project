import torch.nn as nn
import pickle

from utils import seed_everything, set_device, get_train_val_dataloaders, train
from model import FeedForwardRegressionNet

from torch.utils.data import TensorDataset
from torch.optim import SGD, Adam

HIDDEN_DIM = 32  # When using a one layer NN.
LEARNING_RATE = 0.01

BATCH_SIZE = 32
VAL_SIZE = 0.1

device = set_device()

X = pickle.load(open("preprocessed_data/X_train_tensor.pkl", "rb")).float()
y = pickle.load(open("preprocessed_data/y_train_tensor.pkl", "rb")).float()

dataset = TensorDataset(X, y)

train_dataloader, validation_dataloader = get_train_val_dataloaders(
    dataset, VAL_SIZE, BATCH_SIZE
)

input_dim = X.shape[1]
output_dim = 1
model = FeedForwardRegressionNet(input_dim, HIDDEN_DIM, output_dim)

# optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

epochs = 10
criterion = nn.MSELoss()

seed_everything(42)

training_stats = train(
    model,
    epochs,
    train_dataloader,
    validation_dataloader,
    criterion,
    optimizer,
    device=device,
    SSY=y.var() * len(y),
)

