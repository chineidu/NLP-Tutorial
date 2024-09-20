from datetime import datetime

import numpy as np
import torch
from custom_tokenizers import RegexTokenizer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

date_scaler: MinMaxScaler = MinMaxScaler(clip=False)
amount_scaler: MinMaxScaler = MinMaxScaler(clip=False)

# Usage
# dates_array: np.ndarray = date_scaler.fit_transform(dates_array)
# amounts_array: np.ndarray = amount_scaler.fit_transform(amounts_array)


class StatementDataset(Dataset):
    """
    A dataset class for financial statements.

    Attributes
    ----------
    DATE_PAD : int
        Padding value for dates.
    AMOUNT_PAD : float
        Padding value for amounts.
    data : List[List[str]]
        List of transactions, where each transaction is a list of strings.
    labels : List[int]
        List of labels corresponding to each set of transactions.
    tokenizer : RegexTokenizer
        Tokenizer object for encoding descriptions.
    max_length : int
        Maximum length of encoded descriptions.
    max_transactions : int
        Maximum number of transactions to consider.
    """

    DATE_PAD: int = 0
    AMOUNT_PAD: float = 0.0

    def __init__(
        self,
        data: list[list[str]],
        labels: list[int],
        tokenizer: RegexTokenizer,
        max_length: int = 100,
        max_transactions: int = 100,
    ) -> None:
        """
        Initialize the StatementDataset.

        Parameters
        ----------
        data : list[list[str]]
            list of transactions, where each transaction is a list of strings.
        labels : list[int]
            list of labels corresponding to each set of transactions.
        tokenizer : Any
            Tokenizer object for encoding descriptions.
        max_length : int, optional
            Maximum length of encoded descriptions. Default is 100.
        max_transactions : int, optional
            Maximum number of transactions to consider. Default is 100.
        """
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_transactions = max_transactions

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing:
                - 'dates': torch.Tensor of shape (max_transactions,)
                - 'input_ids': torch.Tensor of shape (max_transactions, max_length)
                - 'amounts': torch.Tensor of shape (max_transactions,)
                - 'label': torch.Tensor of shape (1,)

        """
        delimiter: str = " || "
        # Truncate if too many
        transactions: list[str] = self.data[idx][: self.max_transactions]
        label: int = self.labels[idx]

        dates: list[float] = []
        descriptions: list[str] = []
        amounts: list[float] = []

        for transaction in transactions:
            date_str, desc, amount_str = transaction.split(delimiter)
            date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            dates.append(date.timestamp())
            descriptions.append(desc.strip())
            amounts.append(float(amount_str.strip()))

        # Pad if not enough transactions
        while len(dates) < self.max_transactions:
            dates.append(self.DATE_PAD)
            descriptions.append("")  # Empty string for padding descriptions
            amounts.append(self.AMOUNT_PAD)

        # Tokenize descriptions
        encoded: list[list[int]] = self.tokenizer.batch_encode(descriptions, self.max_length)

        # Scale dates and amounts
        scaled_amounts: np.ndarray = amount_scaler.transform(
            np.array(amounts).reshape(-1, 1)
        ).reshape(-1)
        scaled_dates: np.ndarray = date_scaler.transform(np.array(dates).reshape(-1, 1)).reshape(-1)

        return {
            "dates": torch.tensor(scaled_dates, dtype=torch.float32),
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "amounts": torch.tensor(
                scaled_amounts,
                dtype=torch.float32,
            ),
            "label": torch.tensor(label, dtype=torch.long),
        }
