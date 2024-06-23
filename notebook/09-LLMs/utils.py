import os
import sys
from typing import Any

import numpy as np
import tiktoken
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


def go_up_from_current_directory(*, go_up: int = 1) -> None:
    """This is used to up a number of directories.

    Args:
    -----
    go_up (int, default=1): This indicates the number of times to go back up
    from the current directory.

    Returns:
    --------
    None
    """

    CONST: str = "../"
    NUM: str = CONST * go_up

    # Goto the previous directory
    prev_directory = os.path.join(os.path.dirname(__name__), NUM)
    # Get the 'absolute path' of the previous directory
    abs_path_prev_directory = os.path.abspath(prev_directory)

    # Add the path to the System paths
    sys.path.insert(0, abs_path_prev_directory)
    print(abs_path_prev_directory)  # noqa


class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer: Any, max_length: int, stride: int):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        print(f"{len(token_ids) = :,}")  # noqa

        for idx in range(0, len(token_ids) - max_length, stride):
            input_chunk: list[int] = token_ids[idx : (idx + max_length)]
            target_chunk: list[int] = token_ids[idx + 1 : (idx + max_length + 1)]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        x = self.input_ids[idx]
        y = self.target_ids[idx]
        return (x, y)


def create_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a dataloader for the given text data."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset: Dataset = GPTDataset(
        text=text, tokenizer=tokenizer, max_length=max_length, stride=stride
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # self.mask: Create a mask to prevent attention to the future tokens
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Batch size, sequence length, input dimension
        b, num_tokens, d_in = x.shape

        queries: Tensor = self.W_query(x)
        keys: Tensor = self.W_key(x)
        values: Tensor = self.W_value(x)
        # Switch the last 2 dimensions
        attn_scores: Tensor = queries @ keys.transpose(-1, -2)
        # Inplace operation
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float("-inf"))
        attn_weights: Tensor = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector: Tensor = attn_weights @ values

        return context_vector


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "`d_out` should be divisible by `num_heads`"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        # self.mask: Create a mask to prevent attention to the future tokens
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Batch size, sequence length, input dimension
        b, num_tokens, d_in = x.shape
        queries: Tensor = self.W_query(x)
        keys: Tensor = self.W_key(x)
        values: Tensor = self.W_value(x)

        # Reshape and transpose the data
        # Split the matrix by adding `num_heads` dimension and unroll the last dimension
        # i.e. [b, num_tokens, d_out] -> [b, num_tokens, num_heads, head_dim]
        # and transpose the 2nd and 3rd dimensions
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores: Tensor = torch.matmul(queries, keys.transpose(-2, -1))
        # Truncate mask and apply to attention scores
        mask_bool: Tensor = self.mask.bool()[:num_tokens, :num_tokens]
        # Inplace operation: Replace the 1s with -inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute the attention weights
        attn_weights: Tensor = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Transpose to the initial shape:
        # [b, num_heads, num_tokens, head_dim] -> [b, num_tokens, num_heads, head_dim]
        context_vector: Tensor = torch.matmul(attn_weights, values).transpose(1, 2)

        # Combine the heads: Reshape back to [b, num_tokens, d_out]
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        # Apply optional linear output projection
        context_vector = self.out_proj(context_vector)
        return context_vector


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        gelu: Tensor = (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
        return gelu


class FeedForward(nn.Module):
    """Applies a feed-forward neural network to the input tensor `x`. The feed-forward network
    consists of two linear layers with a GELU activation in between. The first linear layer
    expands the input dimension by a factor of 4, and the second linear layer projects the result
    back to the original input dimension.

    Args:
        x (torch.Tensor): The input tensor to be passed through the feed-forward network.

    Returns:
        torch.Tensor: The output tensor after passing through the feed-forward network.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        self.eps: float = 1e-5  # Prevents zero division

        # Trainable params that's automatically adjusted by the LLM during
        # training to improve model performance
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, unbiased=False, keepdim=True)
        norm_x: Tensor = (x - mean) / (std + self.eps)

        return (self.scale * norm_x) + self.shift


# === Variable names are slighly different from the notebook version ===
class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: Tensor) -> Tensor:
        shortcut: Tensor = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        # Add the original input to the output.
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        # Apply shortcut connection.
        x = x + shortcut

        return x


# === Variable names are slighly different from the notebook version ===
class GPTModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()

        # Lookup tables
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape
        tok_embeds: Tensor = self.tok_emb(x)
        pos_embeds: Tensor = self.pos_emb(torch.arange(seq_len, device=x.device))
        # Add token and positional embeddings to get input embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits: Tensor = self.out_head(x)

        return logits


def assign(left: Tensor, right: Tensor) -> nn.Parameter:
    """Assigns the values from the right tensor to the left tensor, ensuring that the
    shapes match.

    Args:
        left (torch.Tensor): The tensor to assign the values to.
        right (torch.Tensor): The values to assign to the left tensor.

    Returns:
        nn.Parameter: The left tensor with the assigned values.

    Raises:
        ValueError: If the shapes of the left and right tensors do not match.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: nn.Module, params: dict[str, Any]) -> None:
    """Assigns the pre-trained weights from the provided parameters to the specified
    layers of the GPT model.

    Args:
        gpt (torch.nn.Module): The GPT model to load the weights into.
        params (dict): A dictionary containing the pre-trained parameter values.

    Raises:
        ValueError: If the shapes of the tensors to be assigned do not match.
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
