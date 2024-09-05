import os
from collections import Counter
from itertools import chain
from typing import Callable, Iterator

import gensim
import spacy
from matplotlib import pyplot as plt
from spacy.tokens import Doc, Token
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_sm")


def spacy_tokenizer(
    corpus: str | list[str],
    batch_size: int = 1_000,
    remove_non_alphanumeric: bool = True,
) -> list[str]:
    """
    Tokenize a corpus of text using spaCy.

    Parameters
    ----------
    corpus : str | list[str]
        A string or list of strings to be tokenized.
    batch_size : int, optional
        The number of texts to process in a single batch, by default 1_000.
    remove_non_alphanumeric : bool, optional
        If True, remove tokens that are not alphanumeric, by default True.

    Returns
    -------
    list[str]
        A list of strings.

    Notes
    -----
    This function uses spaCy to tokenize the input corpus. It applies lowercasing and
    filtering based on the `remove_non_alphanumeric` parameter.
    """
    if isinstance(corpus, str):
        corpus = [corpus]
    tokens: list[str] = []

    # Filtering function
    if remove_non_alphanumeric:
        filter_fn: Callable[[Token], bool] = (
            lambda token: token.text.isalnum() and len(token.text) > 1
        )
    else:
        filter_fn = lambda token: len(token.text) > 1

    # Tokenize the corpus
    doc: Doc = list(nlp.pipe(corpus, disable=["parser", "ner"], batch_size=batch_size))
    tokens.extend([token.text.lower() for sent in doc for token in sent if filter_fn(token)])

    return tokens


def preprocess(doc: str, custom_stopwords: set[str] | None = None) -> list[list[str]]:
    """
    Preprocess a document by tokenizing it and removing stopwords.

    Parameters
    ----------
    doc : str
        The document to preprocess.
    custom_stopwords : set[str] | None, optional
        Custom stopwords to remove, by default None.

    Returns
    -------
    list[list[str]]
        A list of preprocessed tokens.

    Notes
    -----
    This function uses gensim's simple preprocessing to tokenize the input document,
    and removes any tokens that are in the custom_stopwords set.
    """
    if custom_stopwords is None:
        custom_stopwords = set()
    return [word for word in gensim.utils.simple_preprocess(doc) if word not in custom_stopwords]


def spacy_preprocess(doc: str, custom_stopwords: set[str] | None = None) -> list[list[str]]:
    """
    Preprocess the input document using spaCy tokenizer and remove custom stopwords.

    Parameters
    ----------
    doc : str
        The input document to be preprocessed.
    custom_stopwords : set[str] | None, optional
        A set of custom stopwords to be removed from the tokenized document.
        If None, an empty set will be used. Default is None.

    Returns
    -------
    list[list[str]]
        A list of preprocessed tokens with custom stopwords removed.
    """
    if custom_stopwords is None:
        custom_stopwords = set()
    return [word for word in spacy_tokenizer(doc) if word not in custom_stopwords]  # type: ignore


def infer_stopwords(docs: list[list[str]], threshold_percentage: float = 0.5) -> set[str]:
    """
    Infer stopwords from a list of documents based on a frequency threshold.

    Parameters
    ----------
    docs : list[list[str]]
        A list of documents, where each document is represented as a list of tokens.
    threshold_percentage : float, optional
        The percentage threshold for considering a word as a stopword.
        Default is 0.5 (50%).

    Returns
    -------
    set[str]
        A set of inferred stopwords.
    """
    # Flatten the list of documents into a single list of words
    all_words: list[str] = list(chain(*docs))
    word_freq: Counter[str] = Counter(all_words)
    threshold: int = int(len(docs) * threshold_percentage)
    return {word for word, count in word_freq.items() if count > threshold}


def save_tokenized_corpus(tok_corpus: list[list[str]], filepath: str, separator: str) -> None:
    """
    Save a tokenized corpus to a file.

    Parameters
    ----------
    tok_corpus : list[list[str]]
        The tokenized corpus to save. Shape: (n_documents, n_tokens_per_document)
    filepath : str
        The path to the file where the corpus will be saved.
    separator : str
        The separator to use between tokens in each document.

    Returns
    -------
    None

    Notes
    -----
    If the file already exists, the function will not overwrite it and will
    print a message indicating that the save operation was skipped.
    """
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            for doc in tok_corpus:
                f.write(separator.join(doc) + "\n")
        print(f"Saved tokenized corpus to {filepath}")  # noqa: T201
        return
    print(f"File {filepath!r} already exists. Skipping save operation.")  # noqa: T201


class MyCorpus:
    """
    A class to represent a corpus of text data.

    Parameters
    ----------
    filepath : str
        The path to the file containing the corpus data.
    separator : str, optional
        The separator used in the file to separate tokens (default is ",").

    Attributes
    ----------
    filepath : str
        The path to the file containing the corpus data.
    separator : str
        The separator used in the file to separate tokens.
    """

    def __init__(self, filepath: str, separator: str = ",") -> None:
        self.filepath: str = filepath
        self.separator: str = separator

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath!r})"

    def __len__(self) -> int:
        return sum(1 for _ in open(self.filepath, "r"))  # noqa: SIM115

    def __iter__(self) -> Iterator[list[str]]:
        """
        Iterate over the corpus, yielding tokenized lines.

        Yields
        ------
        list[str]
            A list of tokens from each line in the corpus.
        """
        for line in open(self.filepath, "r"):  # noqa: SIM115
            # Corpus has already been converted to lowercase
            yield line.strip().split(self.separator)


def stream_corpus(corpus_iterable: MyCorpus | list[list[str]], size: int = 100) -> list[list[str]]:
    """
    Stream a corpus of tokens from an iterable source.

    Parameters
    ----------
    corpus_iterable : MyCorpus | list[list[str]]
        An iterable containing tokenized documents. Each document is a list of strings.
    size : int, optional
        The number of documents to stream from the corpus, by default 100.

    Returns
    -------
    list[list[str]]
        A list of tokenized documents, where each document is a list of strings.
        Shape: (n_documents, n_tokens_per_document)

    Notes
    -----
    If the corpus is smaller than the specified size, it will return all available documents
    and print a message indicating that the corpus is smaller than the requested size.
    """
    all_tokens: list[list[str]] = []
    try:
        for _ in range(size):
            tokens: list[str] = next(iter(corpus_iterable))
            all_tokens.append(tokens)
    except StopIteration:
        print(f"Corpus is smaller than the size {size}")  # noqa: T201
    return all_tokens


def create_wordcloud(topic_words: str | dict[str, int], title: str) -> None:
    """
    Create and display a word cloud from topic words.

    Parameters
    ----------
    topic_words : str | [dict[str, int]
        Either a dictionary of words and their frequencies, or a string of text.
    title : str
        The title for the word cloud plot.

    Returns
    -------
    None
        This function displays the word cloud plot but does not return any value.

    Notes
    -----
    This function uses matplotlib to display the word cloud.
    """
    wordcloud: WordCloud = WordCloud(width=800, height=500, background_color="white")

    if isinstance(topic_words, dict):
        wordcloud = wordcloud.generate_from_frequencies(topic_words)
    elif isinstance(topic_words, str):
        wordcloud = wordcloud.generate(topic_words)
    else:
        raise ValueError("topic_words must be either a string or a dictionary")

    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title.title(), fontsize=16)
    plt.tight_layout()
    plt.show()
