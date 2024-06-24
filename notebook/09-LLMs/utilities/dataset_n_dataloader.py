from typing import Any

import polars as pl
import torch
from hf_utils import Transformation
from torch import Tensor
from torch.utils.data import Dataset, default_collate
from transformers import (
    AutoTokenizer,
    BatchEncoding,
)

pretrained_model_name_or_path: str = ""
max_length: int = 512
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
# Required for the `tokenization_collate_fn`
transformation = Transformation(tokenizer=tokenizer, max_length=max_length)


class SpendDataset(Dataset):
    def __init__(
        self, data: pl.DataFrame, text_column: str = "Text", label_column: str = "Label"
    ) -> None:
        self.texts: list[str] = [text for text in data.select(text_column).to_series()]
        self.targets: list[int] = data.select(label_column).to_series().to_list()
        assert len(self.texts) == len(self.targets), "Number of texts and targets don't match"

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text: str = self.texts[idx]
        label: Tensor = torch.tensor(self.targets[idx], dtype=torch.long)

        return (text, label)


def collate_fn(batch: list[tuple[str, int]]) -> dict[str, list[Any]]:
    """Collates a batch of text and label pairs into a dictionary of tensors.
    i.e. dynamically pad the inputs received after tokenization.

    Args:
        batch (list[tuple[str, int]]): A list of tuples, where each tuple contains a text string
        and a corresponding label integer.

    Returns:
        dict[str, list]: A dictionary containing the collated text and labels as lists of tensors.
    """
    # Extract the data and labels and create a generator
    texts, labels = zip(*batch)

    # Tokenize the texts
    encoded_inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    # Add labels to the encoded inputs
    encoded_inputs["labels"] = torch.tensor(labels)
    return encoded_inputs


def tokenization_collate_fn(batch: list[tuple[str, int]]) -> tuple[BatchEncoding, Tensor]:
    """
    Collates a batch of tokenized text and label pairs into a BatchEncoding and a Tensor.

    Args:
        batch (list[tuple[str, int]]): List of text strings and label integers.

    Returns:
        tuple[BatchEncoding, Tensor]: BatchEncoding of texts and a Tensor of labels.
    """

    # Extract the data and labels and create a generator
    texts, labels = default_collate(batch)
    # Tokenize the texts with padding and truncation
    encodings: BatchEncoding = transformation(texts)
    return (encodings, labels)
