# ---------------------------------------------------------------
# Chapter 5.5: Loading pretrained weights from OpenAI's GPT-2
# ---------------------------------------------------------------
# The book downloads OpenAI's original TensorFlow checkpoint, which needs
# TensorFlow installed. This uses the book's alternative loader instead:
# the same GPT-2 weights in "safetensors" format from Hugging Face, which
# only needs PyTorch.

import os

import requests
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from gpt_model import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias (OpenAI's GPT-2 uses biases)
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

HF_REPOS = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl",
}


def download_file(url, destination):
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return

    # Download to a temporary name first so an interrupted download
    # never leaves a broken file behind
    tmp_destination = destination + ".part"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        file_size = int(response.headers.get("content-length", 0))
        progress_bar_description = os.path.basename(destination)
        with tqdm(total=file_size, unit="iB", unit_scale=True,
                  desc=progress_bar_description) as progress_bar:
            with open(tmp_destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    file.write(chunk)
                    progress_bar.update(len(chunk))
    os.rename(tmp_destination, destination)


def download_and_load_gpt2(model_name, models_dir="gpt2"):
    if model_name not in HF_REPOS:
        raise ValueError(f"Model name not in {tuple(HF_REPOS)}")

    os.makedirs(models_dir, exist_ok=True)
    repo = HF_REPOS[model_name]
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
    file_path = os.path.join(models_dir, f"{repo.split('/')[-1]}.safetensors")

    download_file(url, file_path)
    params = load_file(file_path)

    # Some copies of the file prefix every name with "transformer."
    return {key.removeprefix("transformer."): value for key, value in params.items()}


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte.weight"])

    for b in range(len(gpt.trf_blocks)):
        # OpenAI stores the query, key and value weights in one matrix
        q_w, k_w, v_w = torch.chunk(
            params[f"h.{b}.attn.c_attn.weight"], 3, dim=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = torch.chunk(
            params[f"h.{b}.attn.c_attn.bias"], 3, dim=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params[f"h.{b}.attn.c_proj.bias"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params[f"h.{b}.mlp.c_proj.bias"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params[f"h.{b}.ln_2.bias"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f.bias"])
    # GPT-2 reuses the token embedding weights in the output layer ("weight tying")
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])


def load_pretrained_gpt2(model_name="gpt2-small (124M)", models_dir="gpt2"):
    """Build a GPTModel and fill it with OpenAI's GPT-2 weights."""
    config = BASE_CONFIG.copy()
    config.update(MODEL_CONFIGS[model_name])

    model = GPTModel(config)
    params = download_and_load_gpt2(model_name, models_dir)
    load_weights_into_gpt(model, params)
    model.eval()
    return model, config
