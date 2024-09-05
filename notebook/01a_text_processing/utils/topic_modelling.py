from collections.abc import Iterable
from typing import List, Tuple

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, Phrases
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({"info": "#76FF7B", "warning": "#FBDDFE", "error": "#FF0000"})
console = Console(theme=custom_theme)


class LdaTopicExtractor:
    """
    A class for performing topic modeling using LDA.

    Steps required to perform topic modeling:
    1. Create a BoW corpus using the corpus_iterable / preprocessed corpus.
    2. Create n-grams from the corpus.
    3. Create an LDA model using the corpus and TF-IDF representation.
    4. Use the LDA model to extract topics and their corresponding words.

    Attributes
    ----------
    corpus : Iterable[List[str]]
        The input corpus for topic modeling.
    num_topics : int
        The number of topics to extract.
    chunksize : int
        The number of documents to be used in each training chunk.
    iterations : int
        Maximum number of iterations through the corpus when inferring the topic
        distribution of a corpus.
    passes : int
        Number of passes through the corpus during training.
    """

    def __init__(
        self,
        corpus: Iterable[List[str]],
        num_topics: int = 10,
        chunksize: int = 2_000,
        iterations: int = 400,
        passes: int = 30,
    ) -> None:
        """
        Initialize the TopicAnalyzer.

        Parameters
        ----------
        corpus : Iterable[List[str]]
            The input corpus for topic modeling.
        num_topics : int, optional
            The number of topics to extract, by default 10.
        chunksize : int, optional
            The number of documents to be used in each training chunk, by default 2000.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution
            of a corpus, by default 400.
        passes : int, optional
            Number of passes through the corpus during training, by default 30.
        """
        self.corpus: Iterable[List[str]] = corpus
        self.num_topics: int = num_topics
        self.chunksize: int = chunksize
        self.iterations: int = iterations
        self.passes: int = passes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_corpus={len(self.corpus):,})"  # type: ignore

    def _create_bigrams(self) -> Phrases:
        """
        Create bigrams from the corpus.

        Returns
        -------
        Phrases
            A Phrases model for creating bigrams.
        """
        return Phrases(sentences=self.corpus, min_count=20)

    def add_bigrams(self) -> Iterable[List[str]]:
        """
        Add bigrams to the corpus.

        Returns
        -------
        Iterable[List[str]]
            The corpus with added bigrams.
        """
        bigram: Phrases = self._create_bigrams()
        for doc in self.corpus:
            yield (doc + [token for token in bigram[doc] if "_" in token])

    def create_bow(self) -> Tuple[Dictionary, List[List[Tuple[int, int]]]]:
        """
        Create a Bag of Words (BoW) representation of the corpus.

        Returns
        -------
        Tuple[Dictionary, List[List[Tuple[int, int]]]]
            A tuple containing the dictionary and the BoW corpus.
        """
        self.corpus_wth_bigrams_: List[List[str]] = list(self.add_bigrams())
        dictionary_: Dictionary = Dictionary(self.corpus_wth_bigrams_)
        corpus_bow: List[List[Tuple[int, int]]] = [
            dictionary_.doc2bow(doc) for doc in self.corpus_wth_bigrams_
        ]
        return dictionary_, corpus_bow

    def train_lda(self) -> LdaModel:
        """
        Train the LDA model.

        Returns
        -------
        LdaModel
            The trained LDA model.
        """
        dictionary_: Dictionary
        corpus_bow: List[List[Tuple[int, int]]]
        dictionary_, corpus_bow = self.create_bow()
        self.lda_model: LdaModel = LdaModel(
            corpus=corpus_bow,
            id2word=dictionary_,
            chunksize=self.chunksize,
            alpha="auto",
            eta="auto",
            iterations=self.iterations,
            num_topics=self.num_topics,
            passes=self.passes,
            eval_every=None,
        )
        console.print("LDA model trained.", style="info")
        return self.lda_model

    def extract_topics(self) -> List[Tuple[List[Tuple[str, float]], float]]:
        """
        Extract topics from the trained LDA model.

        Returns
        -------
        List[Tuple[List[Tuple[str, float]], float]]
            A list of tuples containing the top topics and their coherence scores.
        """
        corpus_bow: List[List[Tuple[int, int]]]
        _, corpus_bow = self.create_bow()
        self.top_topics: List[Tuple[List[Tuple[str, float]], float]] = self.lda_model.top_topics(
            corpus=corpus_bow, topn=self.num_topics
        )
        avg_topic_coherence: float = sum([t[1] for t in self.top_topics]) / self.num_topics
        console.print(f"Average topic coherence: {avg_topic_coherence:.4f}\n", style="info")
        return self.top_topics

    @staticmethod
    def print_topics(top_topics: List[Tuple[List[Tuple[str, float]], float]]) -> None:
        """
        Print the top topics.

        Parameters
        ----------
        top_topics : List[Tuple[List[Tuple[str, float]], float]]
            A list of tuples containing the top topics and their coherence scores.
        """
        console.print(top_topics, style="info")

    @staticmethod
    def _compute_topic_coherence(
        model: LdaModel, corpus_with_bigrams: List[List[str]], dictionary: Dictionary
    ) -> float:
        """
        Compute the topic coherence score.

        Parameters
        ----------
        model : LdaModel
            The trained LDA model.
        corpus_with_bigrams : List[List[str]]
            The corpus with bigrams.
        dictionary : Dictionary
            The dictionary used for creating the BoW corpus.

        Returns
        -------
        float
            The computed coherence score.
        """
        coherence_model: CoherenceModel = CoherenceModel(
            model=model,
            texts=corpus_with_bigrams,
            dictionary=dictionary,
            coherence="u_mass",
        )
        coherence_score: float = coherence_model.get_coherence()
        return coherence_score

    @staticmethod
    def _compute_perplexity(model: LdaModel, corpus_bow: List[List[Tuple[int, int]]]) -> float:
        """
        Compute the perplexity of the model.

        Parameters
        ----------
        model : LdaModel
            The trained LDA model.
        corpus_bow : List[List[Tuple[int, int]]]
            The BoW representation of the corpus.

        Returns
        -------
        float
            The computed perplexity.
        """
        perplexity: float = model.log_perplexity(corpus_bow)
        return perplexity

    def evaluate_lda(self) -> Tuple[float, float]:
        """
        Evaluate the LDA model by computing coherence score and perplexity.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the coherence score and perplexity.
        """
        dictionary_: Dictionary
        corpus_bow: List[List[Tuple[int, int]]]
        dictionary_, corpus_bow = self.create_bow()
        model: LdaModel = self.train_lda()
        coherence_score: float = self._compute_topic_coherence(
            model, self.corpus_wth_bigrams_, dictionary_
        )
        perplexity: float = self._compute_perplexity(model, corpus_bow)
        console.print(
            f"N_topics: {self.num_topics} | Coherence score: {coherence_score:.4f} "
            f"| Perplexity: {perplexity:.4f}",
            style="info",
        )
        return coherence_score, perplexity
