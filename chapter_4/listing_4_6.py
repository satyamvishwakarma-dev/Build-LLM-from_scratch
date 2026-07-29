from chapter_3.listing_3_5 import MultiHeadAttention
from chapter_4.listing_4_3 import FeedForward
from chapter_4.topic_4_1 import GPT_CONFIG_124M
from torch import nn
import torch


class TransformerBLock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_head"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


torch.manual_seed(123)
x = torch.randn(2, 3, 768)
block = TransformerBLock(GPT_CONFIG_124M)
output = block(x)

if __name__ == "__main__":
    print("Input Shape:", x.shape)
    print("\nOutput Shape:", output.shape)
    print("\nOutput:\n", output)
