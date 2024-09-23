import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_metrics(
    pred: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
) -> tuple[float, float]:
    """
    Compute accuracy and average loss for a batch of predictions.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted labels, shape (batch_size,)
    labels : torch.Tensor
        True labels, shape (batch_size,)
    loss : torch.Tensor
        Loss value, shape (1,)

    Returns
    -------
    tuple[float, float]
        Accuracy and average loss
    """
    acc: float = (pred == labels).float().mean().item()
    avg_loss: float = loss.item()
    return acc, avg_loss


def process_batch(
    batch: dict[str, torch.Tensor],
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process a batch of data through the model.

    Parameters
    ----------
    batch : dict[str, torch.Tensor]
        Dictionary containing batch data
    model : nn.Module
        Neural network model
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to run the computations on
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer for model parameters, by default None

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Predicted labels and loss
    """
    dates: torch.Tensor = batch["dates"].to(device)
    input_ids: torch.Tensor = batch["input_ids"].to(device)
    amounts: torch.Tensor = batch["amounts"].to(device)
    labels: torch.Tensor = batch["label"].to(device)

    # Forward pass
    logits: torch.Tensor = model(dates, input_ids, amounts)
    loss: torch.Tensor = criterion(logits, labels)
    pred: torch.Tensor = torch.argmax(logits, dim=1)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return pred, loss


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
    eval_iter: int = 20,
) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    """
    Train a neural network model.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_dataloader : DataLoader
        DataLoader for the training dataset.
    val_dataloader : DataLoader
        DataLoader for the validation dataset.
    num_epochs : int
        Number of epochs to train the model.
    lr : float, optional
        Learning rate for the optimizer, by default 0.001.
    device : torch.device, optional
        Device to run the computations on, by default torch.device("cpu").
    eval_iter : int, optional
        Number of iterations between logging training progress, by default 20.

    Returns
    -------
    tuple[nn.Module, list[float], list[float], list[float], list[float]]
        Trained model, training losses, validation losses, training accuracies,
        validation accuracies.
    """
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    logger.info(f"===== Training for {num_epochs} epochs. =====")

    for epoch in range(num_epochs):
        model.train()

        total_acc: float = 0.0
        total_loss: float = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            pred: torch.Tensor
            loss: torch.Tensor
            pred, loss = process_batch(batch, model, criterion, device, optimizer)
            acc: float
            loss_val: float
            acc, loss_val = compute_metrics(pred, batch["label"].to(device), loss)

            total_acc += acc * len(batch["label"])
            total_loss += loss_val * len(batch["label"])

            if batch_idx % eval_iter == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss_val:.4f}, "
                    f"Accuracy = {acc*100:.2f}%"
                )

        avg_train_acc: float = total_acc / len(train_dataloader.dataset)
        avg_train_loss: float = total_loss / len(train_dataloader.dataset)

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # Validation
        val_acc: float
        val_loss: float
        val_acc, val_loss = evaluate_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(
            f"Epoch: {epoch+1}/{num_epochs} | Train accuracy: {avg_train_acc*100:.2f}% | "
            f"Train loss: {avg_train_loss:.4f}\n"
            f"Val accuracy: {val_acc*100:.2f}% | Val loss: {val_loss:.4f}\n"
        )

    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Evaluate a neural network model on a dataset.

    Parameters
    ----------
    model : nn.Module
        The neural network model to evaluate.
    dataloader : DataLoader
        DataLoader for the dataset to evaluate on.
    criterion : nn.Module
        Loss function to use for evaluation.
    device : torch.device, optional
        Device to run the computations on, by default torch.device("cpu").

    Returns
    -------
    tuple[float, float]
        Average accuracy and average loss on the dataset.
    """
    model.eval()
    total_acc: float = 0.0
    total_loss: float = 0.0

    with torch.no_grad():
        for batch in dataloader:
            pred: torch.Tensor
            loss: torch.Tensor
            pred, loss = process_batch(batch, model, criterion, device)
            acc: float
            loss_val: float
            acc, loss_val = compute_metrics(pred, batch["label"].to(device), loss)

            total_acc += acc * len(batch["label"])
            total_loss += loss_val * len(batch["label"])

    avg_acc: float = total_acc / len(dataloader.dataset)
    avg_loss: float = total_loss / len(dataloader.dataset)

    return avg_acc, avg_loss


def main_training_loop(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-3,
    seed: int = 42,
    eval_iter: int = 20,
) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    """
    Train and evaluate a neural network model.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_dataloader : DataLoader
        DataLoader for the training dataset.
    val_dataloader : DataLoader
        DataLoader for the validation dataset.
    num_epochs : int
        Number of epochs to train the model.
    learning_rate : float, optional
        Learning rate for the optimizer, by default 1e-3.
    seed : int, optional
        Random seed for reproducibility, by default 42.
    eval_iter : int, optional
        Number of iterations between logging training progress, by default 20.

    Returns
    -------
    tuple[nn.Module, list[float], list[float], list[float], list[float]]
        Trained model, training losses, validation losses, training accuracies,
        and validation accuracies.
    """
    torch.manual_seed(seed)

    start_time: float = time.time()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device,
        eval_iter=eval_iter,
    )
    end_time: float = time.time()
    execution_time_minutes: float = (end_time - start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    return model, train_losses, val_losses, train_accs, val_accs
