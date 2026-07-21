import torch

# the input is a vector of length 6
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your  (x^1)
        [0.55, 0.87, 0.66],  # jurney  (x^2)
        [0.57, 0.85, 0.64],  # starts  (x^3)
        [0.22, 0.58, 0.33],  # with  (x^4)
        [0.77, 0.25, 0.10],  # one  (x^5)
        [0.05, 0.80, 0.55],  # step  (x^6)
    ]
)


# the query is a vector of length 1
query = inputs[1]

# creates a attention score for 1D tensors with the batch size of 1
attn_scores_2 = torch.empty(inputs.shape[0])

# iterates over the input tensor and calculates the dot product
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# getting the attention weights by dividing the attention scores by the sum of the attention scores
att_weight_2_tmp = attn_scores_2 / attn_scores_2.sum()


# making the softmax function for the attention weights
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


# using the softmax function to get the attention weights
attn_weight_2_naive = softmax_naive(attn_scores_2)

# using the softmax function of the torch library to get the attention weights
attn_weight_2 = torch.softmax(attn_scores_2, dim=0)

context_vector_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector_2 += x_i * attn_weight_2[i]

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

attn_scores_using_matmul = inputs @ inputs.T

attn_weights = torch.softmax(attn_scores, dim=-1)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])

all_context_vectors = attn_weights @ inputs


if __name__ == "__main__":
    # prints the attention scores
    print(attn_scores_2)

    # prints the attention weights
    print("Attention weights:", att_weight_2_tmp)
    print("Sum of attention weights:", att_weight_2_tmp.sum())

    print("\n===USING NAIVE SOFTMAX===")
    print("Attention weights:", attn_weight_2_naive)
    print("Sum of attention weights:", attn_weight_2_naive.sum())

    print("\n===USING TORCH SOFTMAX===")
    print("Attention weights:", attn_weight_2)
    print("Sum of attention weights:", attn_weight_2.sum())

    print("\n===USING CONTEXT VECTOR===")
    print(context_vector_2)

    print("\n===PRINTING ATTENTION SCORES===")
    print(attn_scores)

    print("\n===PRINTING ATTENTION SCORES USING MATRIX MULTIPLICATION===")
    print(attn_scores_using_matmul)

    print("\n===PRINTING ATTENTION WEIGHTS USING SOFTMAX===")
    print(attn_weights)

    print("\n===NORMALIZED ATTENTION WEIGHTS USING SOFTMAX===")
    print("Row 2 sum:", row_2_sum)
    print("All row sums:", attn_weights.sum(dim=-1))

    print("\n===PRINTING ALL CONTEXT VECTORS===")
    print(all_context_vectors)
