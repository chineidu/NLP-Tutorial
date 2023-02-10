import itertools
import warnings
from typing import Any, Optional

import click
import numpy as np
import scipy
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")


class Sentencizer:
    """This is used to convert a document into a list of sentences.
    It returns sentences."""

    def __init__(self) -> None:
        self.nlp = nlp

    def __call__(self, doc: str, *args: Any, **kwargs: Any) -> list[str]:
        # Tokenize
        doc = nlp(doc)
        sentences = list(doc.sents)
        tokenized_sentences = [str(sentence) for sentence in sentences]
        return tokenized_sentences


def load_text_data(*, filepath) -> list[str]:
    """This returns the data as a list of sentences."""

    with open(filepath, "r") as f:
        data = [line.strip() for line in f.readlines()]
    return data


def preprocess_data(*, input_data: list[str]) -> list[str]:
    """This is used to convert the data into sentences.It returns
    the document as a string and as a list of sentences.

    Params:
        input_data (str): The cleaned text data.

    Returns:
        data_str, sentences: The cleaned text data and the sentences.
    """
    # Create the document
    data_str = "".join(input_data)

    # Extact and tokenize the sentences
    sents = Sentencizer()
    sentences = sents(doc=data_str)
    return sentences


def calculate_tfidf(sentences: list[str], stopwords: list[str]) -> scipy.sparse._csr.csr_matrix:
    """This calculates the TFIDF of the data.

    Params:
        sentences (list[str]): List of sentences.
        stopwords (list[str]): List of words that do not add value to the corpus.

    Returns:
        X_transformed (list[str]): The loades stopwords.
    """

    tfidf = TfidfVectorizer(stop_words=stopwords, norm="l1")
    # Calculate TFIDF data
    X_transformed = tfidf.fit_transform(sentences)

    return X_transformed


def load_stop_words(add_words: Optional[list[str]] = None) -> list[str]:
    """This loads spacy stopwords.

    Params:
        add_words (tuple[str]): Additional stopwords to add.

    Returns:
        stopwords (list[str]): The loades stopwords.
    """
    # Load spaCy stopwords
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    stopwords = list(stopwords)
    if add_words:
        stopwords.extend(add_words)
    return stopwords


def _calculate_sentence_score(tfidf_row: np.ndarray) -> float:
    """This returns the average score of the non-zero tfidf value
    for a given sentence."""
    x = tfidf_row[tfidf_row != 0]  # Select the non-zero values
    return x.mean()


def rank_sentences(*, sentences: list[str], stopwords: list[str], num: int = 5) -> None:
    """This ranks and prints out the top 'num' ranked sentences.

     Params:
        sentences (list[str]): List of sentences.
        stopwords (list[str]): List of words that do not add value to the corpus.
        num (int, default=5): The number of sentences to select as the the summary.

    Returns:
        None
    """
    # Calculate TFIDF
    X_transformed = calculate_tfidf(sentences, stopwords)
    # Initialize the score
    scores = np.zeros(len(sentences))

    # Calculate the score for each sentence
    for idx in range(len(sentences)):
        score = _calculate_sentence_score(X_transformed[idx, :])
        scores[idx] = score
    # Sort the scores in descending order
    # and return the sorted indices
    sort_idx = np.argsort(-scores)

    top_idx = sort_idx[:num]
    sorted_idx = top_idx
    top_sentences = [sentences[idx] for idx in top_idx]
    result = tuple(itertools.zip_longest(sorted_idx, top_sentences))

    result = sorted(result, key=lambda x: x[0])
    for _, sent in result:
        print(sent)


@click.command()
@click.option("-f", "--filepath", help="Enter the absolute filepath.")
@click.option("--num", help="Enter the number of sentences to display.", default=0)
def main(filepath: str, num: Optional[int] = None) -> None:
    """This is the main function."""
    text_data = load_text_data(filepath=filepath)
    sentences = preprocess_data(input_data=text_data)
    # Calculate the number of sentences to print
    RATE = 0.40  # Percentage of senteces to display as summary
    if num == 0:  # If num == None
        num = int(len(sentences) * RATE)
    click.echo(f"Number of sentences in input document: {len(sentences)}\n")
    stopwords = load_stop_words()
    rank_sentences(sentences=sentences, stopwords=stopwords, num=num)
    click.echo()


if __name__ == "__main__":
    main()
