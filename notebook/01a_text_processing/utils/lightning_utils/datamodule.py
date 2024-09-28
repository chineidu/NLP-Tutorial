from typing import Any, Dict, Tuple

import lightning as L
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.custom_datasets import StatementDataset


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
