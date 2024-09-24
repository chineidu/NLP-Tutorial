from typing import Any, Iterable, List

import numpy as np
from compress_fasttext.compress import CompressedFastTextKeyedVectors
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.models import FastText, LsiModel, TfidfModel
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from utils import tokenize_by_special_chars  # type: ignore


class TokenizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_digits: bool = True) -> None:
        """
        Parameters
        ----------
        drop_digits : bool, optional
            Whether to remove digits from the text, by default True
        """
        self.drop_digits = drop_digits

    def fit(self, X, y=None) -> "TokenizeTransformer":
        """
        Fit the TokenizeTransformer to the given data.

        Parameters
        ----------
        X : iterable
            The input data to fit the transformer to.
        y : None
            Unused, included for API consistency.

        Returns
        -------
        self : TokenizeTransformer
            The fitted transformer.
        """
        return self

    def transform(self, X: Iterable[str]) -> list[list[str]]:
        """
        Parameters
        ----------
        X : iterable
            The input data to transform.

        Returns
        -------
        list[list[str]]
            A list of lists of strings, where each list of strings is a tokenized
            version of the input text.
        """
        return [tokenize_by_special_chars(text, drop_digits=self.drop_digits) for text in X]


class DictionaryTransformer(BaseEstimator, TransformerMixin):
    """
    Fit a dictionary to the given data and filter out unwanted tokens.

    Parameters
    ----------
    no_below : int, optional
        The minimum number of documents a token must appear in to be kept,
        by default 5
    no_above : float, optional
        The maximum frequency a token can have to be kept, by default 0.45
    tok_length : int, optional
        The minimum length a token must have to be kept, by default 2

    Attributes
    ----------
    dictionary : gensim.corpora.Dictionary | None
        The fitted dictionary object
    """

    def __init__(self, no_below: int = 5, no_above: float = 0.45, tok_length: int = 2) -> None:
        self.no_below: int = no_below
        self.no_above: float = no_above
        self.tok_length: int = tok_length
        self.dictionary: Dictionary | None = None

    def fit(self, X: list[list[str]], y: Any = None) -> "DictionaryTransformer":
        """
        Fit the dictionary to the given data and filter out unwanted tokens.

        Parameters
        ----------
        X : list of list of str
            The input data to fit the dictionary to. Each inner list represents a document.
        y : Any, optional
            Ignored. Kept for scikit-learn API compatibility.

        Returns
        -------
        DictionaryTransformer
            The fitted transformer.
        """
        self.dictionary = Dictionary(X)
        short_token_ids: list[int] = [
            id for tok, id in self.dictionary.token2id.items() if len(tok) <= self.tok_length
        ]
        self.dictionary.filter_tokens(bad_ids=short_token_ids)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        self.dictionary.compactify()
        return self

    def transform(self, X: list[list[str]]) -> dict[str, Any]:
        """
        Transform the input data using the fitted dictionary.

        Parameters
        ----------
        X : list of list of str
            The input data to transform. Each inner list represents a document.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the fitted dictionary and the transformed corpus.
            - 'dictionary': The fitted gensim.corpora.Dictionary object
            - 'corpus': A list of lists, where each inner list contains
            (token_id, token_count) tuples
        """
        data: dict[str, Any] = {}
        data["dictionary"] = self.dictionary
        data["corpus"] = [self.dictionary.doc2bow(doc) for doc in X]  # type: ignore
        return data


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies TF-IDF (Term Frequency-Inverse Document Frequency) to a corpus.

    This transformer fits a TfidfModel to the input corpus and transforms documents
    using the fitted model.

    Attributes
    ----------
    model : TfidfModel | None
        The fitted TF-IDF model.
    """

    def __init__(self) -> None:
        """
        Initialize the TfidfTransformer.
        """
        self.model: TfidfModel | None = None

    def fit(self, X: dict[str, Any], y: Any = None) -> "TfidfTransformer":
        """
        Fit the TF-IDF model to the given corpus.

        Parameters
        ----------
        X : dict[str, Any]
            A dictionary containing the corpus to fit the model to.
            Expected to have a 'corpus' key with the value being the input corpus.
        y : Any, optional
            Ignored. Kept for scikit-learn API compatibility.

        Returns
        -------
        TfidfTransformer
            The fitted transformer.
        """
        corpus: list[list[tuple[int, int]]] = X["corpus"]
        self.model = TfidfModel(corpus)
        return self

    def transform(self, X: dict[str, Any]) -> dict[str, Any]:
        """
        Transform the input corpus using the fitted TF-IDF model.

        Parameters
        ----------
        X : dict[str, Any]
            A dictionary containing the corpus to transform and the dictionary.
            Expected to have 'corpus' and 'dictionary' keys.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the transformed corpus and the original dictionary.
            - 'dictionary': The original dictionary
            - 'corpus': A list of lists, where each inner list contains
            (token_id, tfidf_score) tuples
        """
        corpus: list[list[tuple[int, int]]] = X["corpus"]
        data: dict[str, Any] = {}
        data["dictionary"] = X["dictionary"]
        data["corpus"] = [self.model[doc] for doc in corpus]  # type: ignore
        return data


class LsiTransformer(BaseEstimator, TransformerMixin):
    """
    Latent Semantic Indexing (LSI) transformer.

    This transformer applies LSI to a given corpus, reducing its dimensionality
    to a specified number of topics.

    Parameters
    ----------
    num_topics : int, optional
        Number of topics to extract. Default is 100.
    random_state : int, optional
        Random seed for reproducibility. Default is 123.

    Attributes
    ----------
    model : LsiModel | None
        The fitted LSI model.
    """

    def __init__(self, num_topics: int = 100, random_state: int = 123) -> None:
        self.num_topics = num_topics
        self.random_state = random_state
        self.model: LsiModel | None = None

    def fit(self, X: dict[str, Any], y: Any = None) -> "LsiTransformer":
        """
        Fit the LSI model to the given corpus.

        Parameters
        ----------
        X : dict[str, Any]
            A dictionary containing the corpus and dictionary.
            Expected to have 'corpus' and 'dictionary' keys.
        y : Any, optional
            Ignored. Kept for scikit-learn API compatibility.

        Returns
        -------
        LsiTransformer
            The fitted transformer.
        """
        corpus: list[list[tuple[int, int]]] = X["corpus"]
        dictionary: Dictionary = X["dictionary"]
        self.model = LsiModel(
            corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            random_seed=self.random_state,
        )
        return self

    def transform(self, X: dict[str, Any]) -> csr_matrix:
        """
        Transform the input corpus using the fitted LSI model.

        Parameters
        ----------
        X : dict[str, Any]
            A dictionary containing the corpus to transform.
            Expected to have a 'corpus' key.

        Returns
        -------
        scipy.sparse.csr_matrix
            The LSI-transformed corpus in CSR matrix format.
            Shape: (n_samples, n_topics)
        """
        corpus: list[list[tuple[int, int]]] = X["corpus"]
        corpus_vec: list[list[tuple[int, float]]] = self.model[corpus]  # type: ignore
        return corpus2csc(corpus_vec).transpose()


class FastTextTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that uses FastText embeddings to convert text documents
    into vector representations.

    Parameters
    ----------
    model : FastText
        The pre-trained FastText model.

    Attributes
    ----------
    model : FastText
        The loaded FastText model.
    """

    def __init__(self, model: FastText) -> None:
        self.model: FastText = model

    def get_embeddings(self, document: list[str]) -> np.ndarray:
        """
        Compute the average embedding for a given document.

        Parameters
        ----------
        document : list[str]
            A list of words representing the document.

        Returns
        -------
        np.ndarray
            The average embedding vector for the document.
            Shape: (embedding_size,)
        """
        word_embeddings: list[np.ndarray] = [self.model.wv[word] for word in document]
        document_embedding: np.ndarray = np.mean(word_embeddings, axis=0)
        return document_embedding

    def fit(self, X: list[str], y: Any | None = None) -> "FastTextTransformer":
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : list[str]
            The input samples, each a string representing a document.
        y : Any | None, optional
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        FastTextTransformer
            The fitted transformer.
        """
        return self

    def transform(self, X: list[list[str]]) -> np.ndarray:
        """
        Transform the input documents into their FastText embeddings.

        Parameters
        ----------
        X : list[list[str]]
            The input samples, each a list of strings representing words in a document.

        Returns
        -------
        np.ndarray
            The FastText embeddings for each input document.
            Shape: (n_samples, embedding_size)
        """
        embeddings: np.ndarray = np.array([self.get_embeddings(doc) for doc in X])
        return embeddings


class QuantizedFastTextTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that uses FastText embeddings to convert text documents
    into vector representations.

    Parameters
    ----------
    model : CompressedFastTextKeyedVectors
        The pre-trained compressed FastText model.

    Attributes
    ----------
    model : CompressedFastTextKeyedVectors
        The loaded compressed FastText model.
    """

    def __init__(self, model: CompressedFastTextKeyedVectors) -> None:
        self.model: CompressedFastTextKeyedVectors = model

    def get_embeddings(self, document: List[str]) -> np.ndarray:
        """
        Compute the average embedding for a given document.

        Parameters
        ----------
        document : List[str]
            A list of words representing the document.

        Returns
        -------
        np.ndarray
            The average embedding vector for the document.
            Shape: (embedding_size,)
        """
        word_embeddings: List[np.ndarray] = [self.model[word] for word in document]
        document_embedding: np.ndarray = np.mean(word_embeddings, axis=0)
        return document_embedding

    def fit(self, X: List[str], y: Any | None = None) -> "QuantizedFastTextTransformer":
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : List[str]
            The input samples, each a string representing a document.
        y : Any | None, optional
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        QuantizedFastTextTransformer
            The fitted transformer.
        """
        return self

    def transform(self, X: List[List[str]]) -> np.ndarray:
        """
        Transform the input documents into their FastText embeddings.

        Parameters
        ----------
        X : List[List[str]]
            The input samples, each a list of strings representing words in a document.

        Returns
        -------
        np.ndarray
            The FastText embeddings for each input document.
            Shape: (n_samples, embedding_size)
        """
        embeddings: np.ndarray = np.array([self.get_embeddings(doc) for doc in X])
        return embeddings
