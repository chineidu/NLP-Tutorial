from datetime import datetime
from typing import Any, Dict, Tuple

import lightning as L
import numpy as np
import torch
from custom_tokenizers import RegexTokenizer, WordPieceTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils import cyclical_encode  # type: ignore


class StatementDataset(Dataset):
    """
    A custom Dataset for handling financial statement data.

    Attributes
    ----------
    DATE_PAD : int
        Padding value for dates.
    AMOUNT_PAD : float
        Padding value for amounts.

    Methods
    -------
    __len__()
        Returns the number of items in the dataset.
    __getitem__(idx)
        Returns a dictionary of tensors for a specific index.
    """

    DATE_PAD: int = 2000
    AMOUNT_PAD: float = 0.0

    def __init__(
        self,
        data: list[list[str]],
        labels: list[int],
        tokenizer: WordPieceTokenizer | RegexTokenizer,
        scaler: dict[str, MinMaxScaler | StandardScaler],
        num_tokens: int = 30,
        max_transactions: int = 200,
        year_month_day: bool = False,
    ) -> None:
        """
        Initialize the StatementDataset.

        Parameters
        ----------
        data : list[list[str]]
            List of transactions, where each transaction is a list of strings.
        labels : list[int]
            List of labels corresponding to each set of transactions.
        tokenizer : WordPieceTokenizer | RegexTokenizer
            Tokenizer to use for encoding descriptions.
        scaler : dict[str, MinMaxScaler | StandardScaler]
            Dictionary of scalers for normalizing dates and amounts.
        num_tokens : int, optional
            Maximum number of encoded descriptions, by default 30.
        max_transactions : int, optional
            Maximum number of transactions to consider, by default 200.
        year_month_day : bool, optional
            Whether to include year, month, and day in the encoded descriptions, by default False.
        """
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.num_tokens = num_tokens
        self.max_transactions = max_transactions
        self.year_month_day = year_month_day

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

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing:
            - 'dates': torch.Tensor of shape (max_transactions,)
            - 'input_ids': torch.Tensor of shape (max_transactions, num_tokens)
            - 'amounts': torch.Tensor of shape (max_transactions,)
            - 'label': torch.Tensor of shape (1,)
        """
        delimiter: str = " || "
        # Truncate if too many
        transactions: list[str] = self.data[idx][: self.max_transactions]
        label: int = self.labels[idx]

        dates: list[float] | list[tuple[int, int, int]] = []  # type: ignore
        years: list[float] = []
        months: list[float] = []
        days: list[float] = []
        descriptions: list[str] = []
        amounts: list[float] = []

        for transaction in transactions:
            date_str, desc, amount_str = transaction.split(delimiter)
            date: datetime = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            dates.append(date)  # type: ignore
            descriptions.append(desc.strip())
            amounts.append(float(amount_str.strip()))

        # Pad if not enough transactions
        while len(amounts) < self.max_transactions:
            dates.append(datetime(self.DATE_PAD, 1, 1))  # type: ignore
            descriptions.append("")  # Empty string for padding descriptions
            amounts.append(self.AMOUNT_PAD)

        # Tokenize descriptions
        encoded: list[list[int]] = self.tokenizer.batch_encode(descriptions, self.num_tokens)

        # Scale dates and amounts
        scaled_amounts: np.ndarray = (
            self.scaler["amount_scaler"].transform(np.array(amounts).reshape(-1, 1)).reshape(-1)
        )
        # Process dates
        if self.year_month_day:
            years = np.array([d.year for d in dates])  # type: ignore
            months = np.array([d.month for d in dates])  # type: ignore
            days = np.array([d.day for d in dates])  # type: ignore

            # Assuming a N-year cycle
            sin_year, cos_year = cyclical_encode(years, 300)
            sin_month, cos_month = cyclical_encode(months, 12)
            sin_day, cos_day = cyclical_encode(days, 31)

            date_features = np.column_stack(
                [
                    sin_year,
                    cos_year,
                    sin_month,
                    cos_month,
                    sin_day,
                    cos_day,
                ]
            )
        else:
            timestamps = np.array([d.timestamp() for d in dates]).reshape(-1, 1)  # type: ignore
            date_features = self.scaler["date_scaler"].transform(timestamps)

        return {
            "dates": torch.tensor(date_features, dtype=torch.float32),
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "amounts": torch.tensor(
                scaled_amounts,
                dtype=torch.float32,
            ),
            "label": torch.tensor(label, dtype=torch.long),
        }


class StatementDatasetWithLabels(Dataset):
    """
    A custom Dataset for handling financial statement data.

    Attributes
    ----------ß
    DATE_PAD : int
        Padding value for dates.
    AMOUNT_PAD : float
        Padding value for amounts.
    TEXT_PAD : str
        Padding value for text.

    Methods
    -------
    __len__() -> int
        Returns the number of items in the dataset.
    __getitem__(idx: int) -> dict[str, torch.Tensor]
        Returns a dictionary of tensors for a specific index.
    get_clusters(data: list[str]) -> np.ndarray
        Returns clusters for the given data.
    """

    DATE_PAD: int = 2000
    AMOUNT_PAD: float = 0.0
    TEXT_PAD: str = "[PAD]"

    def __init__(
        self,
        data: list[list[str]],
        labels: list[tuple[int]],
        tokenizer: WordPieceTokenizer | RegexTokenizer,
        scaler: dict[str, MinMaxScaler | StandardScaler],
        num_tokens: int = 30,
        max_transactions: int = 200,
        year_month_day: bool = False,
    ) -> None:
        """
        Initialize the StatementDataset.

        Parameters
        ----------
        data : list[list[str]]
            List of transactions, where each transaction is a list of strings.
        labels : list[tuple[int]]
            List of labels for each transaction.
        tokenizer : WordPieceTokenizer | RegexTokenizer
            Tokenizer to use for encoding descriptions.
        scaler : dict[str, MinMaxScaler | StandardScaler]
            Dictionary of scalers for normalizing dates and amounts.
        num_tokens : int, optional
            Maximum number of encoded descriptions, by default 30.
        max_transactions : int, optional
            Maximum number of transactions to consider, by default 200.
        year_month_day : bool, optional
            Whether to include year, month, and day in the encoded descriptions, by default False.
        """
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.num_tokens = num_tokens
        self.max_transactions = max_transactions
        self.year_month_day = year_month_day

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

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing:
            - 'dates': torch.Tensor of shape (max_transactions, date_features)
            - 'input_ids': torch.Tensor of shape (max_transactions, num_tokens)
            - 'amounts': torch.Tensor of shape (max_transactions,)
            - 'label': torch.Tensor of shape (1,)
        """
        delimiter: str = " || "
        # Truncate if too many
        transactions: list[str] = self.data[idx][: self.max_transactions]
        label: tuple[int] = self.labels[idx]

        dates: list[datetime] = []
        descriptions: list[str] = []
        amounts: list[float] = []

        for transaction in transactions:
            date_str, desc, amount_str = transaction.split(delimiter)
            date: datetime = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            dates.append(date)
            descriptions.append(desc.strip())
            amounts.append(float(amount_str.strip()))

        # Pad if not enough transactions
        while len(amounts) < self.max_transactions:
            dates.append(datetime(self.DATE_PAD, 1, 1))
            descriptions.append(self.TEXT_PAD)
            amounts.append(self.AMOUNT_PAD)

        # Tokenize descriptions
        encoded: list[list[int]] = self.tokenizer.batch_encode(descriptions, self.num_tokens)

        # Scale dates and amounts
        scaled_amounts: np.ndarray = (
            self.scaler["amount_scaler"].transform(np.array(amounts).reshape(-1, 1)).reshape(-1)
        )
        # Process dates
        if self.year_month_day:
            years: np.ndarray = np.array([d.year for d in dates])
            months: np.ndarray = np.array([d.month for d in dates])
            days: np.ndarray = np.array([d.day for d in dates])

            # Assuming a 200-year cycle
            sin_year, cos_year = cyclical_encode(years, 200)
            sin_month, cos_month = cyclical_encode(months, 12)
            sin_day, cos_day = cyclical_encode(days, 31)

            date_features: np.ndarray = np.column_stack(
                [
                    sin_year,
                    cos_year,
                    sin_month,
                    cos_month,
                    sin_day,
                    cos_day,
                ]
            )
        else:
            timestamps: np.ndarray = np.array([d.timestamp() for d in dates]).reshape(-1, 1)
            date_features = self.scaler["date_scaler"].transform(timestamps)

        return {
            "dates": torch.tensor(date_features, dtype=torch.float32),
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "amounts": torch.tensor(
                scaled_amounts,
                dtype=torch.float32,
            ),
            "label": torch.tensor(label, dtype=torch.float32),
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
            self.dataset_config["num_tokens"],
            self.dataset_config["max_transactions"],
            self.dataset_config["year_month_day"],
        )
        self.test_dataset: StatementDataset = StatementDataset(
            test_data,
            test_labels,
            self.dataset_config["tokenizer"],
            self.dataset_config["scaler"],
            self.dataset_config["num_tokens"],
            self.dataset_config["max_transactions"],
            self.dataset_config["year_month_day"],
        )
        self.val_dataset: StatementDataset = StatementDataset(
            val_data,
            val_labels,
            self.dataset_config["tokenizer"],
            self.dataset_config["scaler"],
            self.dataset_config["num_tokens"],
            self.dataset_config["max_transactions"],
            self.dataset_config["year_month_day"],
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
