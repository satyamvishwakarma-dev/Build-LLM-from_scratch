import torch
import tiktoken
from chapter_5.topic_5_1 import model, GPT_CONFIG_124M
from chapter_4.listing_4_8 import generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


tokenizer = tiktoken.get_encoding("gpt2")

inputs = torch.tensor(
    [[16833, 3626, 6100], [40, 1107, 588]]  # ["every effort moves",
)  #  "I printed"]

targets = torch.tensor(
    [[3626, 6100, 345], [1107, 588, 11311]]  # [" effort moves you",
)  #  " printed text"]

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary

token_ids = torch.argmax(probas, dim=-1)

if __name__ == "__main__":

    print(probas.shape)  # Shape: (batch_size, num_tokens, vocab_size)
    print("Token IDs:\n", token_ids)
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0], tokenizer)}")
