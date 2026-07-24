import torch

from listing_3_3 import CausalAttention, batch
import torch.nn as nn


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 1

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)


context_vec = mha(batch)

if __name__ == "__main__":
    print("Context Vector Shape:", context_vec.shape)
    print("\nContext Vector:\n", context_vec)