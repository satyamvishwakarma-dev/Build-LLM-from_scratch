import os
import tiktoken
import torch
from chapter_5.topic_5_1 import model, GPT_CONFIG_124M
from chapter_4.listing_4_8 import generate_text_simple
from chapter_2.listing_6 import create_dataloader_v1


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

logits_flat = logits.flatten(0, 1)
target_flat = targets.flatten()

loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)


# Get the directory of the current script, go up one level, then into chapter_2
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "chapter_2", "the-verdict.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


if __name__ == "__main__":
    # print("Output text: \n", token_ids_to_text(token_ids, tokenizer))
    # print("\n")
    # print("Probas Shape: ", probas.shape)
    # print("\n")
    # print("Token IDs: \n", token_ids)
    # print("\n")
    # print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    # print("\n")
    # print(f"Output batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    # print("\n")
    # print(f"Target probas 1: {target_probas_1}")
    # print("\n")
    # print(f"Target probas 2: {target_probas_2}")
    # print("\n")
    # print("Log Probas: ", log_probas)
    # print("\n")
    # print(f"Average log probas: {avg_log_probas}")
    # print("\n")
    # print(f"Negative average log probas: {neg_avg_log_probas}")
    # print("\n")
    # print("Logits shape: ", logits.shape)
    # print("Targets shape: ", targets.shape)
    # print("\n")
    # print("Flatten logits: ", logits_flat.shape)
    # print("Flatten targets: ", target_flat.shape)
    # print("\n")
    # print(f"Loss: {loss}")
    print("\n")
    print(f"Total characters: {total_characters}")
    print(f"Total tokens: {total_tokens}")
    print("\n")
    print("Train Loader: ")
    for x, y in train_loader:
        print((x.shape, y.shape))
    print("\n")
    print("Validation Loader: ")
    for x, y in val_loader:
        print((x.shape, y.shape))
    print("\n")
