from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
from matplotlib import pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from spacy.tokens import Doc, Token

nlp = spacy.load("en_core_web_sm")


def calculate_class_weights(n_samples: int, total_samples: int, n_classes: int) -> float:
    return np.round(total_samples / (n_samples * n_classes), 4)


import time

from sklearn.model_selection import StratifiedKFold


def train_model_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    n_splits: int = 5,
) -> tuple[Any, list[float], float, float]:
    """
    Train a model using cross-validation and return performance metrics.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target labels array of shape (n_samples,).
    estimator : Any
        The machine learning model to be trained and evaluated.
    n_splits : int, optional
        Number of splits for cross-validation, by default 5.

    Returns
    -------
    tuple[Any, list[float], float, float]
        A tuple containing:
        - The trained estimator
        - List of accuracy scores for each fold
        - Mean accuracy across all folds
        - Standard deviation of accuracy across all folds
    """
    start_time: float = time.time()
    kfold: StratifiedKFold = StratifiedKFold(n_splits=n_splits).split(X, y)

    scores: list[float] = []

    for k, (train, test) in enumerate(kfold):
        estimator.fit(X[train], y[train])
        score: float = estimator.score(X[test], y[test])
        scores.append(score)
        print(f"Fold: {k+1:2d} | Class dist.: {np.bincount(y[train])} | Acc: {score:.3f}")  # noqa

    mean_accuracy: float = np.mean(scores)
    std_accuracy: float = np.std(scores)
    stop_time: float = time.time()
    print(f"\nCV accuracy: {mean_accuracy:.3f} +/- {std_accuracy:.3f}")  # noqa
    print(f"\nTime taken: {stop_time - start_time:.3f} seconds")  # noqa

    return estimator, scores, mean_accuracy, std_accuracy


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], cmap: str = "Set3"
) -> None:
    """
    Create and plot a confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels, shape (n_samples,)
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,)
    labels : list[str]
        List of label names
    cmap : str, optional
        Colormap for the heatmap, by default "Set3"

    Returns
    -------
    None

    Notes
    -----
    This function creates a confusion matrix from the true and predicted labels,
    and then plots it as a heatmap using seaborn.
    """
    # Create the confusion matrix.
    cm: np.ndarray = confusion_matrix(y_true, y_pred)

    # Plot confusion_matrix.
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(cm, annot=True, cmap=cmap, fmt="d", xticklabels=labels, yticklabels=labels)
    ax.set_yticklabels(labels, rotation=0)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def calculate_precision_recall_curves(
    train_class: np.ndarray,
    test_class: np.ndarray,
    y_proba_train: np.ndarray,
    y_proba_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Calculate precision-recall curves and average precision scores for train and test data.

    Parameters
    ----------
    train_class : np.ndarray
        True labels for training data, shape (n_samples,)
    test_class : np.ndarray
        True labels for test data, shape (n_samples,)
    y_proba_train : np.ndarray
        Predicted probabilities for training data, shape (n_samples,)
    y_proba_test : np.ndarray
        Predicted probabilities for test data, shape (n_samples,)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]
        Precision and recall values for train and test data, and their average precision scores
        (precision_train, recall_train, precision_test, recall_test, ap_train, ap_test)
    """
    # Calculate precision-recall curves
    precision_train, recall_train, _ = precision_recall_curve(train_class, y_proba_train)
    precision_test, recall_test, _ = precision_recall_curve(test_class, y_proba_test)

    # Calculate average precision scores
    ap_train: float = average_precision_score(train_class, y_proba_train)
    ap_test: float = average_precision_score(test_class, y_proba_test)

    return precision_train, recall_train, precision_test, recall_test, ap_train, ap_test


def plot_precision_recall_curves(
    recall_train: np.ndarray,
    precision_train: np.ndarray,
    recall_test: np.ndarray,
    precision_test: np.ndarray,
    ap_train: float,
    ap_test: float,
) -> None:
    """
    Plot precision-recall curves for train and test data.

    Parameters
    ----------
    recall_train : np.ndarray
        Recall values for training data, shape (n_thresholds,)
    precision_train : np.ndarray
        Precision values for training data, shape (n_thresholds,)
    recall_test : np.ndarray
        Recall values for test data, shape (n_thresholds,)
    precision_test : np.ndarray
        Precision values for test data, shape (n_thresholds,)
    ap_train : float
        Average precision score for training data
    ap_test : float
        Average precision score for test data

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 8))
    plt.plot(
        recall_train,
        precision_train,
        color="blue",
        lw=2,
        label=f"Train (Avg Precision = {ap_train:.2f})",
    )
    plt.plot(
        recall_test,
        precision_test,
        color="darkorange",
        lw=2,
        label=f"Test (Avg Precision = {ap_test:.2f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def calculate_roc_curve(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve and AUC score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels. Shape (n_samples,)
    y_pred : np.ndarray
        Target scores. Shape (n_samples,)

    Returns
    -------
    fpr : np.ndarray
        False positive rate. Shape (>2,)
    tpr : np.ndarray
        True positive rate. Shape (>2,)
    roc_auc : float
        Area Under the Curve (AUC) score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc: float = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curves(
    fpr_train: np.ndarray,
    tpr_train: np.ndarray,
    roc_auc_train: float,
    fpr_test: np.ndarray,
    tpr_test: np.ndarray,
    roc_auc_test: float,
) -> None:
    """
    Plot ROC curves for train and test sets.

    Parameters
    ----------
    fpr_train : np.ndarray
        False positive rate for train set. Shape (>2,)
    tpr_train : np.ndarray
        True positive rate for train set. Shape (>2,)
    roc_auc_train : float
        AUC score for train set.
    fpr_test : np.ndarray
        False positive rate for test set. Shape (>2,)
    tpr_test : np.ndarray
        True positive rate for test set. Shape (>2,)
    roc_auc_test : float
        AUC score for test set.

    Returns
    -------
    None
    """
    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr_train,
        tpr_train,
        color="blue",
        lw=2,
        label=f"Train ROC curve (AUC = {roc_auc_train:.2f})",
    )
    plt.plot(
        fpr_test,
        tpr_test,
        color="darkorange",
        lw=2,
        label=f"Test ROC curve (AUC = {roc_auc_test:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_learning_curve(
    estimator: object,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    cv: int = 5,
) -> None:
    """
    Plot the learning curve for a given estimator.

    Parameters
    ----------
    estimator : object
        The machine learning model to evaluate.
    X : np.ndarray of shape (n_samples, n_features)
        The input samples.
    y : np.ndarray of shape (n_samples,)
        The target values.
    train_sizes : np.ndarray of shape (n_points,), default=np.linspace(0.1, 1.0, 10)
        The points of the learning curve to evaluate.
    cv : int, default=5
        The number of folds in cross-validation.

    Returns
    -------
    None
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        # n_jobs=1,
    )

    train_mean: np.ndarray = np.mean(train_scores, axis=1)
    train_std: np.ndarray = np.std(train_scores, axis=1)
    test_mean: np.ndarray = np.mean(test_scores, axis=1)
    test_std: np.ndarray = np.std(test_scores, axis=1)

    plt.figure(figsize=(6, 6))
    plt.plot(
        train_sizes,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training accuracy",
    )

    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        train_sizes,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation accuracy",
    )

    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.grid()
    plt.xlabel("Number of training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.5, 1.03])
    plt.tight_layout()
    plt.show()


# ==== SpaCy Helper Functions ====
# Register the 'lemma' extension
Token.set_extension("lemma", default=None, force=True)


def custom_lemmatizer(token: Token) -> str:
    """
    Custom lemmatizer function for removing the 'ing' suffix from
    gerunds/present participles.

    Parameters
    ----------
    token : Token
        A spaCy Token object.

    Returns
    -------
    str
        The lemmatized form of the token.
    """
    if token.tag_ == "VBG":  # Check if it's a gerund/present participle
        return token.lemma_.rstrip("ing")  # Remove 'ing' from the end of the lemma
    return token.lemma_  # Default to spaCy's lemmatizer for other cases


# Set the custom lemmatizer as a getter for the 'custom_lemma' extension
Token.set_extension("custom_lemma", getter=custom_lemmatizer, force=True)


def spacy_tokenize(text: str | list[str], lemmatize: bool = True) -> list[list[str]]:
    """
    Tokenize and optionally lemmatize text using spaCy.

    Parameters
    ----------
    text: str | list[str]
        Input text or list of texts to process.
    lemmatize : bool, optional
        Whether to lemmatize tokens, by default True.

    Returns
    -------
    list[list[str]]
        A list of lists, where each inner list contains tokens or lemmas for a sentence.

    Notes
    -----
    The shape of the output array is (n_sentences, n_tokens_per_sentence).
    """
    if isinstance(text, str):
        text = [text]
    my_doc: list[Doc] = list(nlp.pipe(text, disable=["parser", "ner"]))

    result: list[list[str]] = []
    for sent in my_doc:
        if lemmatize:  # use the custom lemmatizer
            result.append([token._.custom_lemma for token in sent])
        else:
            result.append([token.text for token in sent])

    return result
