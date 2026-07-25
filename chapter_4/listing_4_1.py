import token

import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embedding = nn.Embedding(
            config["context_length"], config["emb_dim"]
        )
        self.drop_embedding = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(config) for _ in range(config["n_layer"])]
        )
        self.final_norm = DummyLayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_index):
        batch_size, seq_len = in_index.size()
        token_embedding = self.token_embedding(in_index)
        position_embedding = self.position_embedding(
            torch.arange(seq_len, device=in_index.device)
        )
        x = token_embedding + position_embedding
        x = self.drop_embedding(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
