from typing import Any

import tiktoken
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


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
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores: Tensor = torch.matmul(queries, keys.transpose(-2, -1))
        # Truncate mask and apply to attention scores
        mask_bool: Tensor = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute the attention weights
        attn_weights: Tensor = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector: Tensor = torch.matmul(attn_weights, values).transpose(1, 2)

        # Combine the heads
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        # Apply optional linear output projection
        context_vector = self.out_proj(context_vector)
        return context_vector
