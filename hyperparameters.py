import random

search_space = {
    "num_layers": [1, 2, 3, 4],
    "hidden_dim": [4, 8, 16, 64, 128],
    "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "embed_dim0": [4, 8, 16, 32, 64, 128],
    "embed_dim1": [4, 8, 16, 32, 64, 128],
    "epochs": [4, 8, 16, 32],
}

print(random.sample(search_space["num_layers"], 2))


def sample_hyperparameters():
    num_layers = random.sample(search_space["num_layers"], 1)[0]
    return {
        "hidden_dims": random.sample(search_space["hidden_dim"], num_layers),
        "learning_rate": random.sample(search_space["learning_rate"], 1)[0],
        "embed_dim0": random.sample(search_space["embed_dim0"], 1)[0],
        "embed_dim1": random.sample(search_space["embed_dim1"], 1)[0],
        "epochs": random.sample(search_space["epochs"], 1)[0],
    }
