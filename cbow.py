import torch
from torch import nn


class CBOW(nn.Module):
    def __init__(self, num_words, emb_dim):
        super().__init__()

        self.encoder = nn.Embedding(num_words, emb_dim, padding_idx=0)

        self.encoder = nn.Linear(emb_dim, num_words)

    def forward(self, context):
        """
        Predict the center word given the context
        :param context: tensor of shape (batch_size, 2*m)
                m is the context size
        :return: normalized output
        """
        v = self.encoder(context)  # (batch_size, 2*m, emb_dim)
        v = v.mean(dim=1)  # (batch_size, emb_dim)
        z = self.encoder(v)  # (batch_size, num_words)
        y_hat = torch.softmax(z, dim=1)
        return y_hat  # Then optimize y with cross entropy loss
