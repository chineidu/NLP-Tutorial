from typing import Any

import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
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
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits: Tensor = model(input_batch)[:, -1, :]
            predicted_labels: Tensor = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(
    input_batch: Tensor, target_batch: Tensor, model: nn.Module, device: torch.device
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
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # Logits of the final output token
    logits: Tensor = model(input_batch)[:, -1, :]
    loss: Tensor = F.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
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


def train_classifier_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    tokenizer: Any,
) -> tuple[list[float] | int, ...]:
    """
    Trains a simple classifier model on the provided training and validation data.

    Args:
        model (nn.Module): The classifier model to train.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        optimizer (Optimizer): The optimizer to use for training.
        device (torch.device): The device to use for training (CPU or GPU).
        num_epochs (int): The number of training epochs.
        eval_freq (int): The frequency (in steps) to evaluate the model on the validation set.
        eval_iter (int): The number of validation iterations to run.
        tokenizer (Tokenizer): The tokenizer to use for the input data.

    Returns:
        tuple: A tuple containing the following:
            - train_losses (list): The training losses for each evaluation step.
            - val_losses (list): The validation losses for each evaluation step.
            - train_accs (list): The training accuracies for each epoch.
            - val_accs (list): The validation accuracies for each epoch.
            - examples_seen (int): The total number of training examples seen.
    """
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

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

    return train_losses, val_losses, train_accs, val_accs, examples_seen


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

    Args:
        epochs_seen (list): List of epoch numbers.
        examples_seen (list): List of cumulative examples seen.
        train_values (list): List of training values to plot.
        val_values (list): List of validation values to plot.
        label (str, optional): Label for the y-axis and plot title. Defaults to "loss".

    Returns:
        None. The function saves the plot as a PDF and displays it.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

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
