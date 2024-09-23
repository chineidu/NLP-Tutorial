from collections import defaultdict
from typing import Literal, Tuple

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline


def find_optimal_clusters(
    embeddings: np.ndarray, min_clusters: int = 2, max_clusters: int = 5
) -> Tuple[int, float]:
    """
    Find the optimal number of clusters using silhouette score.

    Parameters
    ----------
    embeddings : np.ndarray
        The input embeddings array of shape (n_samples, n_features).
    min_clusters : int, optional
        The minimum number of clusters to try, by default 2.
    max_clusters : int, optional
        The maximum number of clusters to try, by default 5.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the optimal number of clusters and its corresponding
        silhouette score.
    """
    best_k: int = 0
    best_score: float = -1.0

    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_labels: np.ndarray = kmeans.fit_predict(embeddings)
        score: float = silhouette_score(embeddings, cluster_labels, metric="cosine")

        if score > best_score:
            best_score = np.round(score, 4)
            best_k = k
    print(f"best k: {best_k}")  # noqa: T201
    return best_k, best_score


def get_narration_clusters_kmeans(
    embeddings: np.ndarray, n_clusters: int, random_state: int = 42
) -> list[int]:
    max_num_clusters: int = 5
    n_clusters = min(n_clusters, max_num_clusters)

    # Create a KMeans clustering model with 3 clusters
    kmeans = KMeans(n_clusters=n_clusters, max_iter=500, random_state=random_state)
    # Fit the model to your data
    kmeans.fit(embeddings)

    # Get the cluster labels for each data point
    return kmeans.labels_


def get_narration_clusters_dbscan(
    embeddings: np.ndarray, min_samples: int = 3, metric: str = "cosine"
) -> list[int]:
    # Create a DBSCAN clustering model
    dbscan = DBSCAN(eps=0.50, min_samples=min_samples, metric=metric)
    # Fit the model to your data
    dbscan.fit(embeddings)
    # Get the cluster labels for each data point
    return dbscan.labels_


def get_narration_clusters(
    embeddings: np.ndarray,
    n_clusters: int | None | Literal["auto"] = None,
    method: Literal["kmeans", "dbscan"] = "dbscan",
    min_samples: int = 3,
    random_state: int = 42,
) -> list[int]:
    if method not in ["kmeans", "dbscan"]:
        raise ValueError("method must be either 'kmeans' or 'dbscan'")

    if n_clusters is None and method == "kmeans":
        raise ValueError("n_clusters must be specified if method is 'kmeans'")

    if n_clusters is not None and method == "dbscan":
        raise ValueError("n_clusters must not be specified if method is 'dbscan'")

    if n_clusters == "auto":
        n_clusters, _ = find_optimal_clusters(embeddings)
    if method == "kmeans":
        return get_narration_clusters_kmeans(
            embeddings=embeddings,
            n_clusters=n_clusters,  # type: ignore
            random_state=random_state,  # type: ignore
        )
    return get_narration_clusters_dbscan(embeddings=embeddings, min_samples=min_samples)


def display_cluster_results(
    data: pl.DataFrame, tok_df: pl.DataFrame, labels: np.ndarray, pipe: Pipeline
) -> None:
    """
    Process the sample DataFrame, add cluster labels, and display results
    for each cluster.

    Parameters
    ----------
    data : pl.DataFrame
        The input sample DataFrame.
    tok_df : pl.DataFrame
        The token DataFrame to be concatenated.
    labels : np.ndarray, shape (n_samples,)
        Cluster labels for each sample.
    pipe : Pipeline
        A pipeline object with a transform method for text processing.

    Returns
    -------
    None
    """
    data = data.clone()
    data = data.with_columns(cluster=labels)
    data = pl.concat([data, tok_df], how="horizontal").drop(["id", "char_length", "label"])

    for label in np.unique(labels):
        try:
            df: pl.DataFrame = data.filter(pl.col("cluster").eq(label)).sort(
                "date", descending=True
            )
            narr_sim: float = np.mean(
                cosine_similarity(pipe.transform(df["description"].to_list())),
                axis=None,
            )
            print(f"label: {label}")  # noqa: T201
            print(f"transaction similarity: {narr_sim:.4f}")  # noqa: T201
            print(df)  # noqa: T201
            print()  # noqa: T201
        except pl.ShapeError:
            print(f"label with limited data: {label}")  # noqa: T201


def get_clustered_docs(
    data: pl.DataFrame, labels: list[int], pipe: Pipeline
) -> defaultdict[int, tuple[pl.DataFrame, float]]:
    """
    Cluster documents based on given labels and calculate similarity within clusters.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame containing document data.
    labels : list[int]
        List of cluster labels for each document.
    pipe : Pipeline
        A pipeline object with a transform method for text processing.

    Returns
    -------
    defaultdict[int, tuple[pl.DataFrame, float]]
        A dictionary where keys are cluster labels and values are tuples containing:
        - pl.DataFrame: Filtered and sorted DataFrame for the cluster.
        - float: Mean cosine similarity of documents within the cluster.

    Notes
    -----
    This function assumes the existence of a global 'pipe' object for text transformation.
    """

    data = data.clone()
    data = data.with_columns(cluster=labels).drop(["id", "char_length", "label"])
    clustered_data: defaultdict = defaultdict(list)  # type: ignore

    for label in np.unique(labels):
        df: pl.DataFrame = data.filter(pl.col("cluster").eq(label)).sort("date", descending=True)
        narr_sim: float = np.mean(
            cosine_similarity(pipe.transform(df["description"].to_list())),
            axis=None,
        )

        clustered_data[label] = (df, narr_sim)

    return clustered_data
