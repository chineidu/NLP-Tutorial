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
