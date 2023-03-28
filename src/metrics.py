"""This module contains the metrics."""

import itertools
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix


# pylint: disable=too-many-locals
def plot_confusion_matrix(
    *,
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    classes: Union[bool, list[str], None] = None,
    figsize: tuple[int, int] = (12, 12),
) -> None:
    """This returns a confusion matrix plot.
    Params:
      y_true (np.ndarray): The ground truth. i.e the true values
      y_pred (np.ndarray): t=The predicted values.
      classes (Union[bool, List[str], None], default=None): The class label names.
    Returns:
      A Matplotlib Plot
    """
    PCT, SIZE = 100, 12
    WHITE, BLACK = "white", "black"

    # confusion matrix
    c_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # Normalize the values
    c_matrix_norm = np.divide(c_matrix.astype(float), c_matrix.sum(axis=1))
    n_classes = c_matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    # Display an array as a matrix in a new figure window.
    mat = ax.matshow(c_matrix, cmap=plt.cm.Blues)  # pylint: disable=no-member
    # Add a color bar to the side of the plot
    fig.colorbar(mat)

    labels = classes if classes else np.arange(c_matrix.shape[0])

    # Label the axes
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
    )

    # Move the label to the bottom
    ax.xaxis.tick_bottom()

    # Adjust the font size
    ax.xaxis.label.set_size(SIZE)
    ax.yaxis.label.set_size(SIZE)
    ax.title.set_size(SIZE)

    # Set the threshold
    threshold = np.mean((c_matrix.max(), c_matrix.min()))

    # Add text
    # itertools.product: Cartesian product of input iterables.
    # It's equivalent to nested for-loops.
    for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        plt.text(
            i,
            j,
            f"{c_matrix[i, j]} ({c_matrix_norm[i, j] * PCT:.2f}%)",
            horizontalalignment="center",
            color=WHITE if c_matrix[i, j] > threshold else BLACK,
            size=SIZE,
        )
