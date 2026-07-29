from chapter_4.listing_4_6 import TransformerBLock
from chapter_4.topic_4_1 import GPT_CONFIG_124M
import torch
from torch import nn


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.positional_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])

        self.trf_block = nn.Sequential(
            *[TransformerBLock(cfg) for _ in range(cfg["n_layer"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)

        position_embeddings = self.positional_embedding(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = token_embeddings + position_embeddings
        x = self.drop_embedding(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
batch = torch.tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]])

total_params = sum(p.numel() for p in model.parameters())

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())

total_size_bytes = total_params * 4

total_size_mb = total_size_bytes / (1024 * 1024)


out = model(batch)
if __name__ == "__main__":
    print("Input batch:\n", batch)
    print("\nOutput batch shape:", out.shape)
    print("\nOutput batch:\n", out)
    print(f"\nTotal Parameters: {total_params:,}")

    print("\nToken embedding layer shape:", model.token_embedding.weight.shape)
    print("\nOutput layer shape:", model.out_head.weight.shape)

    print(f"\nNumber of trainable parameters "
          f"considering weight tying: {total_params_gpt2:,}")


    print(f"\nTotal Size: {total_size_mb:.2f} MB")
