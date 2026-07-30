import tiktoken
import torch

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
next_token_id = torch.argmax(probas).item()

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()


def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


if __name__ == "__main__":
    print("Output text: \n", token_ids_to_text(token_ids, tokenizer))
    print("\n")
    print(inverse_vocab[next_token_id])  # type: ignore
    print("\n")
    print(inverse_vocab[next_token_id])  # type: ignore
    print("\n")
    print_sampled_tokens(probas)
