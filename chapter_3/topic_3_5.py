from multiprocessing import context

import torch
from listing_3_2 import sa_v2
from topic_3_4 import d_out, d_in, inputs

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1] ** 0.5, dim=-1
)
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))

masked_sample = attn_weights * mask_simple

row_sums = masked_sample.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_sample / row_sums

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)



if __name__ == "__main__":
    print("\n===ATTN WEIGHTS ===")
    print(attn_weights)

    print("\n===MASK SIMPLE ===")
    print(mask_simple)

    print("\n===MASKED SAMPLE ===")
    print(masked_sample)

    print("===ATTANTION WEIGHT MATRIX WITH SUM 1 ===")
    print(masked_simple_norm)

    print("===MASKED MATRIX USING -INF===")
    print(masked)