import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Any, Callable, Generator, Iterable, Iterator, List, Tuple

import gensim
import numpy as np
import polars as pl
import spacy
from gensim.models import Phrases
from matplotlib import pyplot as plt
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
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


def fancy_print(
    object: Any,
    title: str = "Result",
    border_style: str = "bright_green",
    content_style: str | None = None,
    show_type: bool = True,
    expand: bool = False,
    return_panel: bool = False,
) -> Panel | None:
    if isinstance(object, dict):
        content = Table(show_header=False, box=box.SIMPLE)
        for key, value in object.items():
            content.add_row(
                Text(str(key), style="cyan"),
                Text(str(value), style=content_style or "white"),
            )
    elif isinstance(object, (list, tuple)):
        content = Table(show_header=False, box=box.SIMPLE)
        for i, item in enumerate(object):
            content.add_row(
                Text(str(i), style="cyan"),
                Text(str(item), style=content_style or "white"),
            )
    else:
        content = Text(str(object), style=content_style or "white")

    if show_type:
        title = f"{title} ({type(object).__name__})"

    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style=border_style,
        expand=expand,
    )
    if return_panel:
        return panel

    console.print(panel)
    return None


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
    drop_digits: bool = False,
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
    drop_digits : bool, default False
        If True, remove digits from the tokens.
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
    pattern_1: str = r"[\s,./;:?!\\_@#$%^&*=()\d\-]"
    pattern_2: str = r"[\s,./;:?!\\_@#$%^&*=()\-]"
    pattern_3: str = r"(\d+)"
    token_threshold: int = 2

    if custom_stopwords is None:
        custom_stopwords = set()
    if isinstance(corpus, str):
        corpus = [corpus]

    tokens: list[str] | list[list[str]] = []  # type: ignore

    # Tokenizer and Filter functions
    def tok_func(doc: str) -> list[str]:
        """Split on special characters, but retain digits."""
        if drop_digits:
            tokens: list[str] = re.compile(pattern_1).split(doc)
        else:
            tokens = re.compile(pattern_2).split(doc)
            tokens = [
                tok for string_ in tokens for tok in re.compile(pattern_3).split(string_) if tok
            ]
        return tokens

    filter_func = (
        lambda tok: tok.strip() and tok not in custom_stopwords and len(tok) > token_threshold
    )

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


def extract_single_data(data: str) -> Tuple[List[str], List[str], List[float]]:
    """
    Extract date, description, and amount from a single data string.

    Parameters
    ----------
    data : str
        A string containing date, description, and amount separated by ' || '.

    Returns
    -------
    Tuple[List[str], List[str], List[float]]
        A tuple containing lists of dates, descriptions, and amounts.
    """
    delimiter: str = " || "

    dates: List[str] = []
    descriptions: List[str] = []
    amounts: List[float] = []

    # Split on the pipe character
    for row in data:
        if row.strip():
            date, description, amount = row.split(delimiter)
            dates.append(date)
            descriptions.append(description)
            amounts.append(float(amount))

    return dates, descriptions, amounts


def extract_all_data(
    data: List[str],
) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Extract date, description, and amount from multiple data strings.

    Parameters
    ----------
    data : List[str]
        A list of strings, each containing date, description, and
        amount separated by ' || '.

    Returns
    -------
    Tuple[List[List[str]], List[List[str]], List[List[float]]]
        A tuple containing lists of lists for dates, descriptions, and amounts.
    """
    dates: List[List[str]] = []
    descriptions: List[List[str]] = []
    amounts: List[List[float]] = []

    for row in data:
        dates_, descriptions_, amounts_ = extract_single_data(row)

        dates.append(dates_)
        descriptions.append(descriptions_)
        amounts.append(amounts_)

    return dates, descriptions, amounts


def convert_date_to_unix_timestamp(date_string: str, format: str = "%Y-%m-%d") -> int:
    """
    Convert a date string to a Unix timestamp.

    Parameters
    ----------
    date_string : str
        The date string to convert.
    format : str, optional
        The format of the date string (default is "%Y-%m-%d").

    Returns
    -------
    int
        The Unix timestamp corresponding to the input date string.
    """
    try:
        if isinstance(date_string, str):
            # Convert the date string to a datetime object
            date_object = datetime.strptime(date_string, format)
        # Convert the datetime object to a Unix timestamp
        unix_timestamp = int(date_object.timestamp())
        return unix_timestamp

    except ValueError as err:
        console.print(f"Error: {err} | date_string: {date_string}")
        return 0


format_1: str = "%Y-%m-%d %H:%M:%S"
format_2: str = "%Y-%m-%dT%H:%M:%SZ"
output_format: str = "%Y-%m-%d"


def parse_date(date_str: str, input_format: str) -> str | None:
    try:
        parsed_date = datetime.strptime(date_str, input_format)
        return parsed_date.strftime(output_format)
    except ValueError:
        return None


def process_dataframe(raw_json: dict[str, Any]) -> pl.DataFrame:
    try:
        new_df: pl.DataFrame = pl.DataFrame(
            raw_json["bankStatement"]["content"]["statement"],
        )
    except KeyError as e:
        raise ValueError(f"Invalid JSON structure: {e}")

    try:
        new_df = (
            new_df.filter(pl.col("type").eq("credit"))
            .rename({"narration": "description"})
            .with_columns(id=pl.lit("1"))
        )
    except pl.exceptions.ColumnNotFoundError as e:
        raise ValueError(f"Required column not found: {e}")

    # format_1
    new_df = new_df.with_columns(
        parsed_date=pl.col("date").map_elements(lambda x: parse_date(x, format_1))
    )

    if new_df["parsed_date"].null_count() == 0:
        return new_df.drop("date").rename({"parsed_date": "date"}).sort("date", descending=True)

    # format_2
    new_df = new_df.with_columns(
        parsed_date=pl.col("date").map_elements(lambda x: parse_date(x, format_2))
    )

    if new_df["parsed_date"].null_count() == 0:
        return new_df.drop("date").rename({"parsed_date": "date"}).sort("date", descending=True)

    # else
    console.print("[WARNING]: Unable to parse dates. Sorting by original date string.")
    return new_df.drop("parsed_date").sort("date", descending=True)


def extract_year_month_day(date_str: str) -> tuple[int, int, int]:
    """
    Extract year, month, and day from a date string.
    """
    date: list[str] = date_str.split("-")
    year, month, day = int(date[0]), int(date[1]), int(date[2])
    return (year, month, day)


def cyclical_encode(data: np.ndarray, max_val: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform cyclical encoding on the given data.

    Parameters
    ----------
    data : np.ndarray
        Input data to be encoded.
    max_val : float
        Maximum value for normalization.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - The sine of the encoded data.
        - The cosine of the encoded data.

    Notes
    -----
    The shape of the output arrays will be the same as the input `data` array.
    """
    data = 2 * np.pi * (data / max_val)
    return np.sin(data), np.cos(data)


def convert_to_unique_labels(text: list[str]) -> str:
    text = ", ".join(text)  # type: ignore
    text = sorted(set(text.split(", ")))  # type: ignore
    text = ", ".join(text)  # type: ignore
    return text  # type: ignore


def create_id_text_mapping(data: pl.DataFrame, with_labels: bool = True) -> pl.DataFrame:
    """
    Create a mapping of IDs to text and labels from the input DataFrame.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame containing columns: analysis_id, date, description, amount, label.
    with_labels : bool, optional
        If True, include the labels in the mapping. If False, only include the
        unique text in the mapping. The default is True.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: id, statement, label.

    Notes
    -----
    The resulting DataFrame will have one row per unique ID, with concatenated text
    and sorted, unique labels.
    """

    if not with_labels:
        columns: list[str] = ["id", "date", "description", "amount"]
        df: pl.DataFrame = data.select(columns).sort("date", descending=True)
        df_grpby: pl.DataFrame = df.group_by("id").agg(
            text=pl.concat_str(
                pl.col("date"),
                pl.col("description"),
                pl.col("amount"),
                separator=" || ",
            )
        )

    else:
        columns: list[str] = ["id", "date", "description", "amount", "label"]  # type: ignore
        df = data.select(columns).sort("date", descending=True)
        df_grpby = (
            df.group_by("id")
            .agg(
                text=pl.concat_str(
                    pl.col("date"),
                    pl.col("description"),
                    pl.col("amount"),
                    separator=" || ",
                ),
                label=pl.col("label").list.join(", ").unique().sort(),
            )
            .with_columns(
                tag=pl.col("label").map_elements(convert_to_unique_labels),
            )
            .with_columns(
                tag=pl.when(pl.col("tag").str.contains("No-Income"))
                .then(pl.lit("No-Income"))
                .otherwise(pl.col("tag"))
            )
        )

    return df_grpby
