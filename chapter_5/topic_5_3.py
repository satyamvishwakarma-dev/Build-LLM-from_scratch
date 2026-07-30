import tiktoken
import torch
from matplotlib import pyplot as plt

from chapter_5.listing_5_1 import GPT_CONFIG_124M, generate_text_simple, model
from chapter_5.listing_5_3 import text_to_token_ids, token_ids_to_text

model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
)

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.15, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.29, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)

# Argmax sampling
next_token_id_argmax = torch.argmax(probas).item()

# Multinomial sampling
torch.manual_seed(123)
next_token_id_multinomial = torch.multinomial(probas, num_samples=1).item()


def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    # Added minlength so bincount covers all vocab keys even if some get 0 counts
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(vocab))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


temperature = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperature]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperature):
    rects = ax.bar(
        x + i * bar_width,
        scaled_probas[i],
        bar_width,
        label=f"Temperature = {T}",
    )
ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()  # type: ignore

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits,
)

topk_probas = torch.softmax(new_logits, dim=0)

if __name__ == "__main__":
    print("\n")
    print(inverse_vocab[next_token_id_argmax])  # Argmax result # type: ignore
    print("\n")
    print(inverse_vocab[next_token_id_multinomial])  # Multinomial result # type: ignore
    print("\n")
    print_sampled_tokens(probas)
    print("\n")
    # plt.show()
    print("\n")
    print("Top logits:", top_logits)
    print("Top positions:", top_pos)
    print("\n")
    print(new_logits)
    print("\n")
    print(topk_probas)
    