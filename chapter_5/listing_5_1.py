import tiktoken
import torch
from chapter_5.topic_5_1 import model, GPT_CONFIG_124M
from chapter_4.listing_4_8 import generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.flatten()  # Flattens any 2D or 3D tensor down to a 1D list
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

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))

avg_log_probas = torch.mean(log_probas)

neg_avg_log_probas = -avg_log_probas * -1

logits_flat = logits.flatten(0,1)
target_flat = targets.flatten()

loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)

if __name__ == "__main__":
    print("Output text: \n", token_ids_to_text(token_ids, tokenizer))
    print("\n")
    print("Probas Shape: ", probas.shape)
    print("\n")
    print("Token IDs: \n", token_ids)
    print("\n")
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print("\n")
    print(f"Output batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    print("\n")
    print(f"Target probas 1: {target_probas_1}")
    print("\n")
    print(f"Target probas 2: {target_probas_2}")
    print("\n")
    print("Log Probas: ",log_probas)
    print("\n")
    print(f"Average log probas: {avg_log_probas}")
    print("\n")
    print(f"Negative average log probas: {neg_avg_log_probas}")
    print("\n")
    print("Logits shape: ", logits.shape)
    print("Targets shape: ", targets.shape)
    print("\n")
    print("Flatten logits: ", logits_flat.shape)
    print("Flatten targets: ", target_flat.shape)
    print("\n")
    print(f"Loss: {loss}")