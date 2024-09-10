import os
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable, Generator, Iterable, Iterator

import gensim
import spacy
from gensim.models import Phrases
from matplotlib import pyplot as plt
from rich.console import Console
from rich.theme import Theme
from spacy.tokens import Doc, Token
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_sm")
custom_theme = Theme(
    {
        "white": "#FFFFFF",  # Bright white
        "info": "#00FF00",  # Bright green
        "warning": "#FFD700",  # Bright gold
        "error": "#FF1493",  # Deep pink
        "success": "#00FFFF",  # Cyan
        "highlight": "#FF4500",  # Orange-red
    }
)

console = Console(theme=custom_theme)


def spacy_tokenizer(
    corpus: str | list[str],
    n_process: int = 1,
    batch_size: int = 1_000,
    remove_non_alphanumeric: bool = True,
) -> list[str]:
    """
    Tokenize a corpus of text using spaCy.

    Parameters
    ----------
    corpus : str | list[str]
        A string or list of strings to be tokenized.
    n_process : int, optional
        The number of processes to use for parallel processing, by default 1.
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
    doc: Doc = list(
        nlp.pipe(
            corpus,
            disable=["parser", "ner"],
            n_process=n_process,
            batch_size=batch_size,
        )
    )
    tokens.extend([token.text.lower() for sent in doc for token in sent if filter_fn(token)])

    return tokens


def tokenize_by_special_chars(
    corpus: str | list[str],
    custom_stopwords: set[str] | None = None,
    flatten: bool = True,
) -> list[str] | list[list[str]]:
    """
    Tokenize the input corpus by splitting on special characters and digits. It retains the digits
    in the tokens and optionally flattens the result.

    Parameters
    ----------
    corpus : str or list of str
        The input text or list of texts to be tokenized.
    custom_stopwords : set of str, optional
        A set of custom stopwords to be removed from the tokens.
    flatten : bool, default True
        If True, return a flat list of tokens. If False, return a list of token lists for
        each input document.

    Returns
    -------
    list of str or list of list of str
        Tokenized and filtered words from the input corpus.
        If flatten is True, returns a flat list of tokens.
        If flatten is False, returns a list of token lists for each input document.
    """
    pattern_1: str = r"[\s,./;:?!\\_@#$%^&*()\-]"
    pattern_2: str = r"(\d+)"

    if custom_stopwords is None:
        custom_stopwords = set()
    if isinstance(corpus, str):
        corpus = [corpus]

    tokens: list[str] | list[list[str]] = []  # type: ignore

    # Tokenizer and Filter functions
    def tok_func(doc: str) -> list[str]:
        # Split on special characters, but retain digits
        tokens: list[str] = re.compile(pattern_1).split(doc)

        tokens = [tok for string_ in tokens for tok in re.compile(pattern_2).split(string_) if tok]
        return tokens

    filter_func = lambda tok: tok.strip() and tok not in custom_stopwords

    for doc in corpus:
        if flatten:
            # Tokenize and remove empty tokens
            tokens.extend(tok_func(doc))  # type: ignore
            tokens = [tok.lower() for tok in tokens if filter_func(tok)]  # type: ignore
        else:
            # Tokenize and remove empty tokens
            doc_tok: list[str] = tok_func(doc)
            doc_tok = [tok.lower() for tok in doc_tok if filter_func(tok)]
            tokens.append(doc_tok)  # type: ignore

    return tokens


def preprocess(
    doc: str,
    custom_stopwords: set[str] | None = None,
) -> list[list[str]]:
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


def spacy_preprocess(
    doc: str,
    n_process: int = 1,
    batch_size: int = 1_000,
    custom_stopwords: set[str] | None = None,
) -> list[str]:
    """
     Preprocess the input document using spaCy tokenizer and remove custom stopwords.

     Parameters
     ----------
     doc : str
         The input document to be preprocessed.
     n_process : int, optional
         The number of processes to use for parallel processing, by default 1.
     batch_size : int, optional
         The number of texts to process in a single batch, by default 1_000.
     custom_stopwords : set[str] | None, optional
         A set of custom stopwords to be removed from the tokenized document.
         If None, an empty set will be used. Default is None.

     Returns
     -------
    list[str]
         A list of preprocessed tokens with custom stopwords removed.
    """
    if custom_stopwords is None:
        custom_stopwords = set()
    return [
        word for word in spacy_tokenizer(doc, n_process, batch_size) if word not in custom_stopwords
    ]


def infer_stopwords(docs: list[list[str]], threshold_percentage: float = 0.4) -> set[str]:
    """
    Infer stopwords from a list of documents based on a frequency threshold.

    Parameters
    ----------
    docs : list[list[str]]
        A list of documents, where each document is represented as a list of tokens.
    threshold_percentage : float, optional
        The percentage threshold for considering a word as a stopword.
        Default is 0.4 (40%).

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


@dataclass
class MyCorpus:
    """
    A class representing a corpus of text data.

    Attributes
    ----------
    input_data : list[Any] | None
        Input data as a list of any type, default is None.
    filepath : str | None
        Path to the file containing the corpus data, default is None.
    separator : str
        Separator used in the file to split the data, default is ",".
    data : list[list[str]]
        The corpus data stored as a list of lists of strings.
    index : int
        Current index for iteration, default is 0.

    Methods
    -------
    load_data_from_file() -> None
        Load the corpus data from the file.
    """

    input_data: list[Any] | None = None
    filepath: str | None = None
    separator: str = ","
    data: list[list[str]] = field(default_factory=list, init=False)
    index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """
        Initialize the corpus data after object creation.

        Raises
        ------
        ValueError
            If neither filepath nor input_data is provided.
        """
        if self.filepath is not None:
            self.load_data_from_file()
        elif self.input_data is not None:
            self.data = self.input_data
        elif self.input_data is None and self.filepath is None:
            raise ValueError("Please provide either a filepath or input_data")

    def load_data_from_file(self) -> None:
        """
        Load the corpus data from the file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if not os.path.exists(self.filepath):  # type: ignore
            raise FileNotFoundError(f"The file {self.filepath} does not exist.")

        with open(self.filepath, "r") as file:  # type: ignore
            self.data = [line.strip().split(self.separator) for line in file]

    def __repr__(self) -> str:
        """
        Return a string representation of the MyCorpus object.

        Returns
        -------
        str
            A string representation of the MyCorpus object.
        """
        source: str = f"filepath={self.filepath!r}" if self.filepath else "in-memory data"
        return f"{self.__class__.__name__}({source}, data={len(self.data):,} items)"

    def __len__(self) -> int:
        """
        Return the number of items in the corpus.

        Returns
        -------
        int
            The number of items in the corpus.
        """
        return len(self.data)

    def __iter__(self) -> Iterator[list[str]]:
        """
        Return an iterator for the corpus.

        Returns
        -------
        Iterator[list[str]]
            An iterator for the corpus.
        """
        self.index = 0
        return self

    def __next__(self) -> list[str]:
        """
        Return the next item in the corpus.

        Returns
        -------
        list[str]
            The next item in the corpus.

        Raises
        ------
        StopIteration
            If there are no more items in the corpus.
        """
        if self.index >= len(self.data):
            raise StopIteration
        value: list[str] = self.data[self.index]
        self.index += 1
        return value


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


def create_bigrams(
    corpus: list[list[str]] | Iterable[list[str]],
    min_count: int = 10,
    threshold: float = 2.0,
) -> Phrases:
    """
    Create a Phrases model from a corpus of documents.

    Parameters
    ----------
    corpus : list[list[str]] | Iterable[list[str]]
        An iterable containing tokenized documents. Each document is a list of strings.
    min_count : int, optional
        Minimum count of n-grams to be included in the output Phrases model,
        by default 10.
    threshold : float, optional
        Threshold for forming the phrases (higher means fewer phrases),
        by default 2.0.

    Returns
    -------
    Phrases
        A Phrases model trained on the given corpus.

    Notes
    -----
    The `min_count` parameter is capped at 0.4 times the length of the
    corpus. This is to prevent the generation of too many phrases.
    """
    min_count = min(min_count, int(0.4 * len(corpus)))  # type: ignore
    return Phrases(sentences=corpus, min_count=min_count, threshold=threshold)


def add_bigrams(
    corpus: list[list[str]] | Iterable[list[str]],
    min_count: int = 10,
    threshold: float = 2.0,
) -> Generator[list[str], None, None]:
    bigram: Phrases = create_bigrams(corpus, min_count, threshold)
    if isinstance(corpus, list):
        corpus = MyCorpus(corpus)

    try:
        # Reset the iterator
        corpus.index = 0  # type: ignore
    except AttributeError:
        console.print("Failed to reset iterator index")
        pass

    for doc in corpus:
        combined = doc + [token for token in bigram[doc] if "_" in token]
        yield combined


class Dataset:
    def __init__(self, data: list[str]):
        """
        Initialize a Dataset object.

        Parameters
        ----------
        data : list[str]
            The list of strings to be iterated over.
        """
        self.data = data
        self.index = 0

    def __repr__(self) -> str:
        """
        Return a string representation of the object.
        """
        return f"{self.__class__.__name__}(data={len(self.data):,} items)"

    def __iter__(self) -> Iterator[list[str]]:
        self.index = 0
        return self

    def __next__(self) -> list[str]:
        """
        Return the next item from the data list, tokenized by special characters.

        Returns
        -------
        list[str]
            The list of tokens for the next item in the data list.

        Raises
        ------
        StopIteration
            If the end of the data list is reached.
        """
        if self.index >= len(self.data):
            raise StopIteration

        value = self.data[self.index]
        self.index += 1
        tok_docs: list[str] = [tok for tok in tokenize_by_special_chars(value)]  # type: ignore
        return tok_docs

    def __len__(self) -> int:
        """
        Return the length of the data list.
        """
        return len(self.data)


def create_wordcloud(
    topic_words: str | dict[str, int], title: str, max_words: int = 50_000
) -> None:
    """
    Create and display a word cloud from topic words.

    Parameters
    ----------
    topic_words : str | [dict[str, int]
        Either a dictionary of words and their frequencies, or a string of text.
    title : str
        The title for the word cloud plot.
    max_words : int, optional
        The maximum number of words to include in the word cloud, by default 50_000

    Returns
    -------
    None
        This function displays the word cloud plot but does not return any value.

    Notes
    -----
    This function uses matplotlib to display the word cloud.
    """
    wordcloud: WordCloud = WordCloud(
        max_words=max_words, width=800, height=600, background_color="white"
    )

    if isinstance(topic_words, dict):
        wordcloud = wordcloud.generate_from_frequencies(topic_words)
    elif isinstance(topic_words, str):
        wordcloud = wordcloud.generate(topic_words)
    else:
        raise ValueError("topic_words must be either a string or a dictionary")

    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title.title(), fontsize=25)
    plt.tight_layout()
    plt.show()
