import torch.nn as nn
import torch
from topic_3_4 import d_in, d_out
from topic_3_3 import inputs

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vectors = attn_weights @ values
        return context_vectors
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)

if __name__ == "__main__":
    print(sa_v1(inputs))