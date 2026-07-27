import torch
from torch import nn
import tiktoken
from listing_4_7 import GPT_CONFIG_124M, model


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
if __name__ == "__main__":
    print("\nEncoded:", encoded)
    print("\nEncoded Tensor:", encoded_tensor)
    print("\nEncoded Tensor Shape:", encoded_tensor.shape)
    print("\nOutput:", out)
    print("\nOutput Length:", len(out[0]))
    print("\nDecoded Text:", decoded_text)
