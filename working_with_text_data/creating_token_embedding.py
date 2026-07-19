import torch

# creatring the tensor as the input
input_ids = torch.tensor([2, 3, 5, 1])

# setting the vocabulary size and output dimension
vocab_size = 6
output_dim = 3


# give the random seed to make the output deterministic
torch.manual_seed(123)

# creating the embedding layer using the vocabulary size and output dimension
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

if __name__ == "__main__":
    # printing the weight of the embedding layer
    print("----- WEIGHT -----")
    print(embedding_layer.weight)
    print("\n")

    # passing the input tensor to the embedding layer
    print("----- OUTPUT -----")
    print(embedding_layer(torch.tensor([3])))
    print("\n")

    # printing the output of the all four input IDs
    print("_____ OUTPUT OF ALL FOUR INPUT IDS _____")
    print(embedding_layer(input_ids))
    print("\n")