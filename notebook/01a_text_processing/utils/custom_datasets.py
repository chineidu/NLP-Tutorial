from datetime import datetime
from typing import Any, Dict, Tuple

import lightning as L
import numpy as np
import torch
from custom_tokenizers import RegexTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

date_scaler: MinMaxScaler = MinMaxScaler(clip=False)
amount_scaler: MinMaxScaler = MinMaxScaler(clip=False)
scaler: dict[str, MinMaxScaler | StandardScaler] = {
    "date_scaler": date_scaler,
    "amount_scaler": amount_scaler,
}

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
    scaler : dict[str, MinMaxScaler | StandardScaler]
        Scaler object for scaling numeric features.
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
        scaler: dict[str, MinMaxScaler | StandardScaler],
        max_length: int = 100,
        max_transactions: int = 200,
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
        scaler : dict[str, MinMaxScaler | StandardScaler]
            Scaler object for scaling numeric features.
        max_length : int, optional
            Maximum length of encoded descriptions. Default is 100.
        max_transactions : int, optional
            Maximum number of transactions to consider. Default is 200.
        """
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.scaler = scaler
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
        scaled_amounts: np.ndarray = (
            self.scaler["amount_scaler"].transform(np.array(amounts).reshape(-1, 1)).reshape(-1)
        )
        scaled_dates: np.ndarray = (
            self.scaler["date_scaler"].transform(np.array(dates).reshape(-1, 1)).reshape(-1)
        )

        return {
            "dates": torch.tensor(scaled_dates, dtype=torch.float32),
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "amounts": torch.tensor(
                scaled_amounts,
                dtype=torch.float32,
            ),
            "label": torch.tensor(label, dtype=torch.long),
        }


class DatasetModule(L.LightningDataModule):
    """
    A LightningDataModule for handling dataset operations.

    Attributes:
        dataset_config (Dict[str, Any]): Configuration dictionary for the dataset.
        batch_size (int): Number of samples per batch.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset_config: Dict[str, Any],
        batch_size: int = 64,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        """
        Initialize the DataSetModule.

        Args:
            dataset_config: Configuration dictionary for the dataset.
            batch_size: Number of samples per batch. Defaults to 64.
            test_size: Proportion of the dataset to include in the test split.
                Defaults to 0.2.
            seed: Random seed for reproducibility. Defaults to 42.
        """
        super().__init__()

        self.dataset_config: Dict[str, Any] = dataset_config
        self.batch_size: int = batch_size
        self.test_size: float = test_size
        self.seed: int = seed

    def prepare_data(self) -> None:
        """Prepare the dataset by downloading the training and test sets from the internet."""
        pass

    def _get_splits(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into train, test, and validation sets.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing train_data, test_data, val_data, train_labels, test_labels,
                val_labels.
                Each element is a numpy array.
        """

        train_data, test_data, train_labels, test_labels = train_test_split(
            self.dataset_config["data"],
            self.dataset_config["labels"],
            test_size=self.test_size,
            stratify=self.dataset_config["labels"],
            random_state=self.seed,
        )
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data,
            train_labels,
            test_size=self.test_size,
            stratify=train_labels,
            random_state=self.seed,
        )
        return train_data, test_data, val_data, train_labels, test_labels, val_labels

    def setup(self, stage: str) -> None:
        """
        Define the setup method which is responsible for loading and splitting the dataset.

        Args:
            stage (str): Current stage ('fit', 'validate', 'test', or 'predict').
        """
        train_data, test_data, val_data, train_labels, test_labels, val_labels = self._get_splits()
        # Create datasets
        self.train_dataset: StatementDataset = StatementDataset(
            train_data,
            train_labels,
            self.dataset_config["tokenizer"],
            self.dataset_config["scaler"],
            self.dataset_config["max_length"],
            self.dataset_config["max_transactions"],
        )
        self.test_dataset: StatementDataset = StatementDataset(
            test_data,
            test_labels,
            self.dataset_config["tokenizer"],
            self.dataset_config["scaler"],
            self.dataset_config["max_length"],
            self.dataset_config["max_transactions"],
        )
        self.val_dataset: StatementDataset = StatementDataset(
            val_data,
            val_labels,
            self.dataset_config["tokenizer"],
            self.dataset_config["scaler"],
            self.dataset_config["max_length"],
            self.dataset_config["max_transactions"],
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create the training data loader.

        Returns:
            DataLoader: The training data loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create the test data loader.

        Returns:
            DataLoader: The test data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed),
        )
