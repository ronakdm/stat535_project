import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardRegressionNet(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size0,
        vocab_size1,
        embed_dim0,
        embed_dim1,
        hidden_dims,
        output_dim,
    ):
        super(FeedForwardRegressionNet, self).__init__()

        # Embedding layers.
        self.embed0 = nn.Embedding(vocab_size0, embed_dim0)
        self.embed1 = nn.Embedding(vocab_size1, embed_dim1)
        self.vocab_size0 = vocab_size0
        self.vocab_size1 = vocab_size1

        # Fully-connected layers.
        hidden_dims.insert(0, embed_dim0 + embed_dim1 + input_dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_dims[i - 1], hidden_dims[i])
                for i in range(1, len(hidden_dims))
            ]
        )
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        embed0 = self.embed0(x[:, 0].int())
        embed1 = self.embed1(x[:, 1].int())
        x_num = x[:, 2:]

        z = torch.cat((embed0, embed1, x_num), dim=1)
        for layer in self.layers:
            z = F.relu(layer(z))
        return z.reshape(-1)
