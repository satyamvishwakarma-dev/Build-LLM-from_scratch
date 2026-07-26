from topic_4_2 import batch_example

from functorch import dim
from torch import nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x  - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)


if __name__ == '__main__':
    print("Mean\n", mean)
    print("\n")
    print("Variance\n", var)