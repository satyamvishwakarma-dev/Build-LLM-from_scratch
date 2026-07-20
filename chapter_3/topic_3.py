import torch

# the input is a vector of length 6
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],     # Your  (x^1)
        [0.55, 0.87, 0.66],     # jurney  (x^2)
        [0.57, 0.85, 0.64],     # starts  (x^3)
        [0.22, 0.58, 0.33],     # with  (x^4)
        [0.77, 0.25, 0.10],     # one  (x^5)
        [0.05, 0.80, 0.55]      # step  (x^6)
    ]
)

# the query is a vector of length 1
query = inputs[1]

# creates a attention score for 1D tensors with the batch size of 1
attn_scores_2 = torch.empty(inputs.shape[0])

# iterates over the input tensor and calculates the dot product
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)


att_weight_2_tmp = attn_scores_2 / attn_scores_2.sum()

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weight_2_naive = softmax_naive(attn_scores_2)

if __name__ == "__main__":
    # prints the attention scores
    print(attn_scores_2)
    print("Attention weights:", att_weight_2_tmp )
    print("Sum of attention weights:", att_weight_2_tmp.sum())
    print("Attention weights:", attn_weight_2_naive)
    print("Sum of attention weights:", attn_weight_2_naive.sum())


