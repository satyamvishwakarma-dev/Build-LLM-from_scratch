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
att_scores_2 = torch.empty(inputs.shape[0])

# iterates over the input tensor and calculates the dot product
for i, x_i in enumerate(inputs):
    att_scores_2[i] = torch.dot(x_i, query)

# prints the attention scores
print(att_scores_2)

