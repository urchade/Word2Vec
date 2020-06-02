import torch
from torch import nn


class Skipgram(nn.Module):
    def __init__(self, num_words, emb_dim, context_size, padding_idx):
        super().__init__()
        self.context_size = context_size

        self.encoder = nn.Embedding(num_words, emb_dim, padding_idx=padding_idx)

        self.encoder = nn.Linear(emb_dim, num_words)

    def forward(self, center_word):
        """
        Predict the context words given the center.
        :param center_word: tensor of shape (batch_size, 2*m)
                m is the context size
        :return: normalized output
        """

        v = self.encoder(center_word)  # (batch_size, emb_dim)
        z = self.encoder(v)  # (batch_size, num_words)
        y_hat = torch.softmax(z, dim=1)

        return y_hat  # Then optimize y_hat given all context words by assuming independence between words
