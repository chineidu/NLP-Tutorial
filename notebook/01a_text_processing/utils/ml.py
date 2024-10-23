# ruff: noqa: T201
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    normalize: Literal["true", "pred", "all"] | None = "true",
    cmap: str = "Set3",
) -> None:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels of shape (n_samples,).
    labels : list[str]
        List of label names.
    normalize : Literal["true", "pred", "all"] | None, optional
        Normalization option for confusion matrix.
        "true" normalizes over the actual (true) conditions.
        "pred" normalizes over the predicted conditions.
        "all" normalizes over all conditions.
        None does not normalize.
        Default is "true".
    cmap : str, optional
        Colormap for the heatmap. Default is "Set3".

    Returns
    -------
    None
        This function plots the confusion matrix and does not return any value.
    """
    # Create the confusion matrix.
    cm: np.ndarray = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Plot confusion_matrix.
    _, ax = plt.subplots(figsize=(8, 5))

    fmt: str = ".2f" if normalize is not None else "d"
    sns.heatmap(cm, annot=True, cmap=cmap, fmt=fmt, xticklabels=labels, yticklabels=labels)
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


def plot_confusion_matrix_multiclass(y_preds, y_true, labels, title: str | None = None) -> None:
    if title is None:
        title = "Normalized confusion matrix".title()
    else:
        title = title.title()

    cm = confusion_matrix(y_true, y_preds, normalize="true")
    _, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(title, size=18)
    plt.tight_layout()
    plt.show()


def plot_multilabel_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    normalize: bool = False,
) -> None:
    """
    Plot multilabel confusion matrices for each label.

    Parameters
    ----------
    y_true : np.ndarray
        True labels, shape (n_samples, n_labels)
    y_pred : np.ndarray
        Predicted labels, shape (n_samples, n_labels)
    label_names : list[str]
        Names of the labels

    Returns
    -------
    None
    """
    # Compute multi-label confusion matrix
    multi_conf_mat: np.ndarray = multilabel_confusion_matrix(y_true, y_pred)
    _, axes = plt.subplots(nrows=len(label_names), ncols=1, figsize=(6, 12))

    for i, (ax, name) in enumerate(zip(axes, label_names)):
        cm: np.ndarray = multi_conf_mat[i]
        if normalize:
            cm = (cm / cm.sum(1)).round(2)
        sns.heatmap(cm, annot=True, fmt=".2f", ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {name}", size=15)
        ax.xaxis.set_ticklabels(["Negative", "Positive"])
        ax.yaxis.set_ticklabels(["Negative", "Positive"])

    plt.tight_layout()
    plt.show()


def print_metrics(multi_conf_mat: np.ndarray, label_names: list[str]) -> None:
    """
    Print performance metrics for each label in a multi-label classification.

    Parameters
    ----------
    multi_conf_mat : np.ndarray
        Multi-label confusion matrix of shape (n_labels, 2, 2).
    label_names : list[str]
        List of label names.

    Returns
    -------
    None
        This function prints the metrics and does not return any value.
    """
    for i, name in enumerate(label_names):
        tn, fp, fn, tp = multi_conf_mat[i].ravel()
        accuracy: float = (tp + tn) / (tp + tn + fp + fn)
        precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1: float = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )

        print(f"\nMetrics for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
    return None


def generate_confusion_matrix_report(
    y_preds: np.ndarray, y_true: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float]]:
    """
    Generate and print a detailed classification report including per-class accuracies
    and overall metrics.

    Parameters
    ----------
    y_preds : np.ndarray
        Predicted labels of shape (n_samples,).
    y_true : np.ndarray
        Ground truth labels of shape (n_samples,).
    labels : list[str]
        List of label names.

    Returns
    -------
    dict[str, dict[str, float]]
        A dictionary containing the classification report with precision, recall,
        f1-score, and support metrics for each class and averages.
    """
    # Generate the classification report
    report = classification_report(y_true, y_preds, target_names=labels, output_dict=True)

    # Calculate accuracy for each class
    cm = confusion_matrix(y_true, y_preds)
    class_accuracies: dict[str, float] = {}
    for i, label in enumerate(labels):
        true_positives = cm[i, i]
        total_instances = np.sum(cm[i, :])
        accuracy = true_positives / total_instances if total_instances > 0 else 0
        class_accuracies[label] = accuracy

    # Print the classification report with accuracies
    print("Classification Report:\n")
    for label in labels:
        print(f"{label}:")
        print("-" * (len(label) + 1))
        print(f"  Precision: {report[label]['precision']:.2f}")
        print(f"  Recall: {report[label]['recall']:.2f}")
        print(f"  F1-Score: {report[label]['f1-score']:.2f}")
        print(f"  Samples: {report[label]['support']:,}")
        print(f"  Accuracy: {class_accuracies[label]:.2f}")
        print()

    print("Overall:")
    print("-" * 9)
    print(f"Accuracy: {report['accuracy']:.2f}")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.2f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.2f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.2f}")
    print(f"Weighted Avg Precision: {report['weighted avg']['precision']:.2f}")
    print(f"Weighted Avg Recall: {report['weighted avg']['recall']:.2f}")
    print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.2f}")
    return report


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
