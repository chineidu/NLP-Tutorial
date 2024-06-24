from typing import Any

import polars as pl
import torch
import torch.nn.functional as F
from data_prep.data_prep import DataCleaner
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding


def _get_num_batches(data_loader: DataLoader, num_batches: int | None = None) -> int:
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    return num_batches


def calc_loss_batch(
    input_batch: BatchEncoding,
    target_batch: Tensor,
    model: nn.Module,
    device: str | torch.device,
) -> Tensor:
    """Calculates the loss for a batch of input and target data using the given model.

    Args:
        input_batch (torch.Tensor): The input batch of data.
        target_batch (torch.Tensor): The target batch of labels.
        model (torch.nn.Module): The model to be used for the loss calculation.
        device (torch.device): The device (CPU or GPU) to use for the calculation.

    Returns:
        torch.Tensor: The loss value for the batch.
    """
    enc_input: dict[str, Any] = {key: value.to(device) for key, value in input_batch.items()}
    target_batch = target_batch.to(device)
    logits: Tensor = model(enc_input)
    loss: Tensor = F.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: str | torch.device,
    num_batches: int | None = None,
) -> float:
    """Calculates the average loss across a data loader for a given model.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader containing the input and
        target data.
        model (torch.nn.Module): The model to be used for the loss calculation.
        device (torch.device): The device (CPU or GPU) to use for the calculation.
        num_batches (int, optional): The maximum number of batches to evaluate. If None,
        all batches in the data loader will be used.

    Returns:
        float: The average loss of the model on the data loader.
    """
    total_loss: float = 0.0

    if len(data_loader) == 0:
        return float("nan")

    num_batches = _get_num_batches(data_loader, num_batches)

    for idx, (input_batch, target_batch) in enumerate(data_loader):
        if idx < num_batches:
            enc_input: dict[str, Any] = {
                key: value.to(device) for key, value in input_batch.items()
            }
            target_batch = target_batch.to(device)
            loss: Tensor = calc_loss_batch(enc_input, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    average_loss: float = total_loss / num_batches

    return average_loss


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: str | torch.device,
    num_batches: int | None = None,
) -> float:
    """Calculates the accuracy of a model on a given data loader.

    Args:
        data_loader (DataLoader): The data loader containing the input data and target labels.
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device (CPU or GPU) to use for the evaluation.
        num_batches (int, optional): The maximum number of batches to evaluate. If None, all
        batches in the data loader will be used.

    Returns:
        float: The accuracy of the model on the data loader.
    """
    model.eval()

    correct_predictions: int = 0
    num_examples: int = 0

    num_batches = _get_num_batches(data_loader, num_batches)

    for idx, (input_batch, target_batch) in enumerate(data_loader):
        if idx < num_batches:
            # Move each element of input_batch to the device
            enc_input: dict[str, Any] = {
                key: value.to(device) for key, value in input_batch.items()
            }
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits: Tensor = model(enc_input)
            predicted_labels: Tensor = torch.argmax(logits, dim=-1)
            correct_predictions += (predicted_labels == target_batch).sum().item()
            num_examples += predicted_labels.shape[0]
        else:
            break
    accuracy: float = correct_predictions / num_examples

    return accuracy


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """
    Evaluates the performance of the given model on the training and validation data.

    Args:
        model (nn.Module): The model to evaluate.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): The device to use for evaluation (CPU or GPU).
        eval_iter (int): The number of evaluation iterations to run.

    Returns:
        tuple: A tuple containing the following:
            - train_loss (float): The average training loss.
            - val_loss (float): The average validation loss.
    """
    model.eval()  # Set the model to evaluation mode.

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()  # Reset the model to training mode.

    return (train_loss, val_loss)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """
    Trains a PyTorch model on the provided training and validation data loaders, and evaluates the
    model's performance.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        device (str | torch.device): The device to use for training and evaluation (CPU or GPU).
        num_epochs (int): The number of training epochs.
        eval_freq (int): The frequency (in steps) at which to evaluate the model during training.
        eval_iter (int): The number of evaluation iterations to run.

    Returns:
        tuple[list[float], list[float], list[float], list[float], int]: A tuple containing:
            - train_losses (list[float]): The training losses for each evaluation step.
            - val_losses (list[float]): The validation losses for each evaluation step.
            - train_accs (list[float]): The training accuracies for each epoch.
            - val_accs (list[float]): The validation accuracies for each epoch.
            - examples_seen (int): The total number of training examples seen.
    """
    # Initialize lists to track losses, accuracies and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss: Tensor = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += len(input_batch["input_ids"])
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

    return (train_losses, val_losses, train_accs, val_accs, examples_seen)


# ====== BONUS ======


def classify_transaction(
    text: str, model: nn.Module, transformation: Any, device: str | torch.device
) -> dict[str, Any]:
    model.eval()

    labels: list[str] = ["bills", "loan", "savingsAndInvestments", "noSpend"]
    cleaned_text: str = (
        DataCleaner(data=pl.DataFrame(data={"text": text}))
        .prepare_data()
        .select(["cleaned_text"])
        .to_series()
        .to_list()[0]
    )

    encoded_input: BatchEncoding = transformation(cleaned_text)
    # Move each element of input_batch to the device
    enc_input: dict[str, Any] = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        logits: torch.Tensor = model(enc_input)
    probas: Tensor = F.softmax(logits, dim=-1).squeeze(0)

    spend_labels: list[tuple[str, int]] = [
        (label, round(proba.item(), 2)) for label, proba in zip(labels, probas)
    ]
    # Sort using the proba
    spend_labels = sorted(spend_labels, key=lambda x: x[1], reverse=True)
    sorted_spend_labels: list[dict[str, float]] = [{label: proba} for label, proba in spend_labels]

    return {"transaction": text, "spend_labels": sorted_spend_labels}
