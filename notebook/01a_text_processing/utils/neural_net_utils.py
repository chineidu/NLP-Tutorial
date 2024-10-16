from typing import Any, Dict, List

import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.custom_datasets import StatementDataset


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """
    Calculates the accuracy of a model on a given data loader.

    Parameters
    ----------
    data_loader : DataLoader
        The data loader containing the input data and target labels.
    model : nn.Module
        The model to be evaluated.
    device : torch.device
        The device (CPU or GPU) to use for the evaluation.
    num_batches : int | None, optional
        The maximum number of batches to evaluate. If None, all
        batches in the data loader will be used.

    Returns
    -------
    float
        The accuracy of the model on the data loader.
    """
    model.eval()
    correct_predictions: int = 0
    num_examples: int = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits: Tensor = model(input_batch)
            predicted_labels: Tensor = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(
    input_batch: Tensor, target_batch: Tensor, model: nn.Module, device: torch.device
) -> Tensor:
    """
    Calculates the loss for a batch of input and target data using the given model.

    Parameters
    ----------
    input_batch : Tensor
        The input batch of data.
    target_batch : Tensor
        The target batch of labels.
    model : nn.Module
        The model to be used for the loss calculation.
    device : torch.device
        The device (CPU or GPU) to use for the calculation.

    Returns
    -------
    Tensor
        The loss value for the batch.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # Logits of the final output token
    logits: Tensor = model(input_batch)
    loss: Tensor = F.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """
    Calculates the average loss across a data loader for a given model.

    Parameters
    ----------
    data_loader : DataLoader
        The data loader containing the input and target data.
    model : nn.Module
        The model to be used for the loss calculation.
    device : torch.device
        The device (CPU or GPU) to use for the calculation.
    num_batches : int | None, optional
        The maximum number of batches to evaluate. If None,
        all batches in the data loader will be used.

    Returns
    -------
    float
        The average loss of the model on the data loader.
    """
    total_loss: float = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss: Tensor = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """
    Evaluates the performance of the given model on the training and validation data.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    train_loader : DataLoader
        The training data loader.
    val_loader : DataLoader
        The validation data loader.
    device : torch.device
        The device to use for evaluation (CPU or GPU).
    eval_iter : int
        The number of evaluation iterations to run.

    Returns
    -------
    tuple[float, float]
        A tuple containing the following:
        - train_loss (float): The average training loss.
        - val_loss (float): The average validation loss.
    """
    model.eval()  # Set the model to evaluation mode.

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()  # Reset the model to training mode.

    return (train_loss, val_loss)


def train_classifier_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
) -> tuple[nn.Module, list[float], list[float], list[float], list[float], int]:
    """
    Train a classifier model and evaluate its performance.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    device : torch.device
        The device to run the model on (CPU or GPU).
    num_epochs : int
        Number of epochs to train the model.
    eval_freq : int
        Frequency of evaluation steps.
    eval_iter : int
        Number of iterations for evaluation.

    Returns
    -------
    tuple[nn.Module, list[float], list[float], list[float], list[float], int]
        A tuple containing:
        - The trained model
        - List of training losses
        - List of validation losses
        - List of training accuracies
        - List of validation accuracies
        - Total number of examples seen during training

    (model, train_losses, val_losses, train_accs, val_accs, examples_seen)
    """
    # Initialize lists to track losses and examples seen
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    examples_seen: int = 0
    global_step: int = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode.

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            # Optional: Evaluate after each batch
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(  # noqa
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Calculate the accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")  # noqa
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")  # noqa
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return model, train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(
    epochs_seen: list[int] | Tensor,
    examples_seen: list[float] | Tensor,
    train_values: list[float] | Tensor,
    val_values: list[float] | Tensor,
    label: str = "loss",
    save_plot: bool = True,
) -> None:
    """
    Plot training and validation values over epochs and examples seen.

    This function creates a plot with two x-axes: one for epochs and another for examples seen.
    It plots both training and validation values for a given label (e.g., "loss" or "accuracy").

    Parameters
    ----------
    epochs_seen : list[int] | Tensor
        List of epoch numbers.
    examples_seen : list[float] | Tensor
        List of cumulative examples seen.
    train_values : list[float] | Tensor
        List of training values to plot.
    val_values : list[float] | Tensor
        List of validation values to plot.
    label : str, optional
        Label for the y-axis and plot title. Defaults to "loss".
    save_plot : bool, optional
        Whether to save the plot as a PDF. Defaults to True.

    Returns
    -------
    None
        The function saves the plot as a PDF and displays it.
    """
    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    if save_plot:
        plt.savefig(f"{label}-plot.pdf")
    plt.show()


# ==== RNN Training ====
# Binary Classification
def train_model_binary(
    dataloader: torch.utils.data.DataLoader, model: nn.Module, lr: float = 0.001
) -> tuple[float, float]:
    """
    Train the model using the provided dataloader.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader containing the training data.
    model : nn.Module
        The neural network model to be trained.
    lr : float, optional
        The learning rate for the optimizer (default is 0.001).

    Returns
    -------
    tuple[float, float]
        A tuple containing the average accuracy and average loss.
    """
    loss_fn: nn.BCELoss = nn.BCELoss()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    total_acc: float = 0
    total_loss: float = 0

    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()

        # Forward pass
        pred: torch.Tensor = model(text_batch, lengths)[:, 0]
        loss: torch.Tensor = loss_fn(pred, label_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)

    avg_acc: float = total_acc / len(dataloader.dataset)
    avg_loss: float = total_loss / len(dataloader.dataset)

    return (avg_acc, avg_loss)


# Binary Classification
def evaluate_model_binary(
    dataloader: torch.utils.data.DataLoader, model: nn.Module
) -> tuple[float, float]:
    """
    Evaluate the model using the provided dataloader.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader containing the evaluation data.
    model : nn.Module
        The neural network model to be evaluated.

    Returns
    -------
    tuple[float, float]
        A tuple containing the average accuracy and average loss.
    """
    loss_fn: nn.BCELoss = nn.BCELoss()

    model.eval()
    total_acc: float = 0
    total_loss: float = 0

    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred: torch.Tensor = model(text_batch, lengths)[:, 0]
            loss: torch.Tensor = loss_fn(pred, label_batch)
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)

    avg_acc: float = total_acc / len(dataloader.dataset)
    avg_loss: float = total_loss / len(dataloader.dataset)
    return (avg_acc, avg_loss)


# ==== RNN Training ====
# Multi-class Classification
def train_model(dataloader: DataLoader, model: nn.Module, lr: float = 0.001) -> tuple[float, float]:
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    total_acc: float = 0
    total_loss: float = 0

    for batch in dataloader:
        dates = batch["dates"]
        input_ids = batch["input_ids"]
        amounts = batch["amounts"]
        labels = batch["label"]

        optimizer.zero_grad()

        # Forward pass
        logits: torch.Tensor = model(dates, input_ids, amounts)
        loss: torch.Tensor = criterion(logits, labels)
        pred: torch.Tensor = torch.argmax(logits, dim=1)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_acc += (pred == labels).float().sum().item()
        total_loss += loss.item() * labels.size(0)

    avg_acc: float = total_acc / len(dataloader.dataset)
    avg_loss: float = total_loss / len(dataloader.dataset)

    return (avg_acc, avg_loss)


# Multi-class Classification
def evaluate_model_multi(
    dataloader: torch.utils.data.DataLoader, model: nn.Module
) -> tuple[float, float]:
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    model.eval()
    total_acc: float = 0
    total_loss: float = 0

    with torch.no_grad():
        for batch in dataloader:
            dates = batch["dates"]
            input_ids = batch["input_ids"]
            amounts = batch["amounts"]
            labels = batch["label"]

            logits: torch.Tensor = model(dates, input_ids, amounts)
            loss: torch.Tensor = criterion(logits, labels)
            pred: torch.Tensor = torch.argmax(logits, dim=1)

            # Update metrics
            total_acc += (pred == labels).float().sum().item()
            total_loss += loss.item() * labels.size(0)

    avg_acc: float = total_acc / len(dataloader.dataset)
    avg_loss: float = total_loss / len(dataloader.dataset)

    return (avg_acc, avg_loss)


def predict_customer_type_batch(
    transactions: List[List[str]],
    model: nn.Module,
    model_dependency: Dict[str, Any],
    max_length: int = 15,
    max_transactions: int = 100,
    year_month_day: bool = False,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Predict customer types for a batch of transactions.

    Parameters
    ----------
    transactions : List[List[str]]
        List of transactions, where each transaction is a list of strings.
    model : nn.Module
        The trained neural network model.
    model_dependency : Dict[str, Any]
        Dictionary containing model dependencies (tokenizer, scaler, label_encoder).
    max_length : int, optional
        Maximum length of each transaction, by default 15.
    max_transactions : int, optional
        Maximum number of transactions to consider, by default 100.
    year_month_day : bool, optional
        Whether to use year-month-day format for dates, by default False.
    batch_size : int, optional
        Batch size for processing, by default 32.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing prediction results:
        - 'predicted': List[str], predicted customer types
        - 'probability': List[List[float]], probabilities for each prediction
        - 'all_labels': List[str], all possible customer type labels
        - 'all_proba': List[List[float]], probabilities for all labels for
        each prediction
    """
    model.eval()

    dataset: StatementDataset = StatementDataset(
        transactions,
        [0] * len(transactions),
        model_dependency["tokenizer"],
        model_dependency["scaler"],
        max_length,
        max_transactions,
        year_month_day=year_month_day,
    )
    le: Any = model_dependency["label_encoder"]
    dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size)

    predicted: List[str] = []
    probas: List[List[float]] = []
    all_probas: List[List[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            logits: torch.Tensor = model(batch["dates"], batch["input_ids"], batch["amounts"])
            proba: torch.Tensor = torch.softmax(logits, dim=1)
            pred: torch.Tensor = torch.argmax(logits, dim=1)

            predicted.extend(le.inverse_transform(pred.cpu().numpy()))
            probas.append(list(proba.max(dim=1).values.cpu().numpy().round(4)))
            all_probas.extend(proba.cpu().numpy())

    return {
        "predicted": predicted,
        "probability": probas,
        "all_labels": le.classes_,
        "all_proba": all_probas,
    }
