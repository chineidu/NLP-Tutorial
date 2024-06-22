# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


# type: ignore
# ruff: noqa
import tiktoken
import torch
from gpt_download import download_and_load_gpt2
from torch import Tensor, nn

# Import from local files
from utils import GPTModel, load_weights_into_gpt


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model: nn.Module, input_idx: Tensor, max_new_tokens: int, context_length: int
) -> Tensor:
    """Generate text using a language model by iteratively predicting the next token.

    Args:
        model (nn.Module): The language model used for generating text.
        input_idx (Tensor): The input tensor containing the initial context tokens.
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_length (int): The maximum length of the context that the model can handle.

    Returns:
        Tensor: The tensor containing the generated sequence of tokens.
    """
    for _ in range(max_new_tokens):
        # Crop the context if it exceeds the maximum length supported by the LLM.
        # i.e. if the LLM supports only 5 tokens and the context is 10 tokens long, reduce
        # it to 5 tokens.
        idx_cond: Tensor = input_idx[:, -context_length:]

        with torch.no_grad():
            logits: Tensor = model(idx_cond)

        # Get the last token from the sequence
        logits = logits[:, -1, :]
        probas: Tensor = torch.softmax(logits, dim=-1)
        idx_next: Tensor = torch.argmax(probas, dim=-1, keepdim=True)
        # Append the last token to the context
        input_idx = torch.cat([input_idx, idx_next], dim=1)

    return input_idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if (
            idx_next == eos_id
        ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def main(gpt_config, input_prompt, model_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    gpt = GPTModel(gpt_config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer),
        max_new_tokens=30,
        context_size=gpt_config["context_length"],
        top_k=1,
        temperature=1.0,
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    torch.manual_seed(123)

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size)
