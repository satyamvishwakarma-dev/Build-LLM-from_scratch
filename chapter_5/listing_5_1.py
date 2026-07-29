import tiktoken
import torch
from chapter_5.topic_5_1 import model, GPT_CONFIG_124M
from chapter_4.listing_4_8 import generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


start_context = "Every effort follows you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])

targets = torch.tensor([[3626, 6100, 345], [1107, 5788, 11311]])

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)


if __name__ == "__main__":
    # Use .squeeze() or .flatten() here so token_ids converts from 3D (2, 3, 1) to a 1D list
    print("Output text: \n", token_ids_to_text(token_ids.squeeze(), tokenizer))
    print("\n")
    print("Probas Shape: ", probas.shape)
    print("Token IDs: \n", token_ids)
    print("\n")
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Output batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")