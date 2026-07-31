import urllib.request

import torch

from chapter_5.gpt_download import download_and_load_gpt2
from chapter_5.listing_5_1 import GPT_CONFIG_124M
from chapter_4.listing_4_7 import GPTModel

# url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"
# filename = url.split("/")[-1]
# urllib.request.urlretrieve(url, filename)

print("Downloading / loading weights from OpenAI...")
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layer": 12, "n_head": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layer": 24, "n_head": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layer": 36, "n_head": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layer": 48, "n_head": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv-bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: Left {left.shape} vs Right {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
