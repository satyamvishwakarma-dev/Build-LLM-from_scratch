############################################################################
# LOADING OPENAI WEIGHTS INTO OUR GPT MODEL CODE
############################################################################

import numpy as np
import torch

from chapter_5.listing_5_4 import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer,
)
from chapter_5.listing_5_3 import device
from chapter_5.topic_5_5 import NEW_CONFIG, GPTModel, download_and_load_gpt2


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: Left {left.shape} vs Right {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


gpt = GPTModel(NEW_CONFIG)
gpt.eval()

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")


def load_weights_into_gpt(gpt, params):
    gpt.positional_embedding.weight = assign(
        gpt.positional_embedding.weight, params["wpe"]
    )
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # Multi-Head Attention Q, K, V Weights
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        # Multi-Head Attention Q, K, V Biases
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        # Output projection
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # Feed Forward / MLP
        gpt.trf_blocks[b].ff.layer[0].weight = assign(
            gpt.trf_blocks[b].ff.layer[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layer[0].bias = assign(
            gpt.trf_blocks[b].ff.layer[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt.trf_blocks[b].ff.layer[2].weight = assign(
            gpt.trf_blocks[b].ff.layer[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layer[2].bias = assign(
            gpt.trf_blocks[b].ff.layer[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # Layer Normalizations (FIXED: scale -> weight, shift -> bias)
        gpt.trf_blocks[b].norm1.weight = assign(
            gpt.trf_blocks[b].norm1.weight, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.bias = assign(
            gpt.trf_blocks[b].norm1.bias, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.weight = assign(
            gpt.trf_blocks[b].norm2.weight, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.bias = assign(
            gpt.trf_blocks[b].norm2.bias, params["blocks"][b]["ln_2"]["b"]
        )

    # Final LayerNorm & Output Head (FIXED: scale -> weight, shift -> bias)
    gpt.final_norm.weight = assign(gpt.final_norm.weight, params["g"])
    gpt.final_norm.bias = assign(gpt.final_norm.bias, params["b"])

    # Output head weight mapping
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you forward", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5,
)

if __name__ == "__main__":
    print("Output text: \n", token_ids_to_text(token_ids, tokenizer))
