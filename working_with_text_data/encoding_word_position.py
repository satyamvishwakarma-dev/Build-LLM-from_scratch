import torch
import listing_6 as l6

# setting the size of the vocabulary and output dimension
vocab_size = 50257
output_dim = 256

# makeing the token embedding layer
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# setting the maximum length of the context
max_length = 4

# creating the dataloader to feed the model
data_loader = l6.create_dataloader_v1(
    l6.raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

# Itrating over the dataloader
data_iter = iter(data_loader)

# getting the first batch of the dataloader
inputs, targets = next(data_iter)

# passing the inputs to the token embedding layer
token_embedding = token_embedding_layer(inputs)

# setting the context length
context_length = max_length

# creating the position embedding layer
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# passing the context length to the position embedding layer
pos_embedding = pos_embedding_layer(torch.arange(context_length))

# adding the position embedding to the token embedding
input_embedding = token_embedding + pos_embedding

if __name__ == "__main__":
    # print("--- INPUTS (What the model sees) ---")
    # print(inputs)
    # print("\n")
    # print("--- SHAPE OF INPUTS ---")
    # print(inputs.shape)
    # print("\n--- TARGETS (What the model predicts) ---")
    # print(targets)
    print("--- SHAPE OF TARGETS ---")
    print(targets.shape)
    print("\n--- SHAPE OF TOKEN EMBEDDING ---")
    print(token_embedding.shape)
    print("\n---SHAPE OF POSITION EMBEDDING ---")
    print(pos_embedding.shape)
    print("\n---SHAPE OF TOKEN EMBEDDING ---")
    print(token_embedding.shape)

