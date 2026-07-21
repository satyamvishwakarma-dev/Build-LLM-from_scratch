from os import name

import torch
from topic_3_3 import inputs

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

key = inputs @ W_key
value = inputs @ W_value

key_2 = key[1]
attn_score_22 = query_2.dot(key_2)

attn_score_2 = query_2 @ key.T

d_k = key.shape[-1]

attn_weight_2 = torch.softmax(attn_score_2 / d_k ** 0.5, dim=-1)

context_value_2 = attn_weight_2 @ value

if __name__ == "__main__":
    print(query_2)

    print("Key Shape:", key.shape)
    print("Value Shape:", value.shape)

    print(attn_score_22)

    print(attn_score_2)

    print(attn_weight_2)

    print(context_value_2)