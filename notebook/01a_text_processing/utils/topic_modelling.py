from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, Phrases
from rich.console import Console
from rich.theme import Theme

from utils.utils import MyCorpus

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
    corpus : MyCorpus
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

    Example
    -------
    >>> corpus: list[list[str]] = [
    ...     ["this", "is", "the", "first", "document"],
    ...     ["this", "is", "the", "second", "document"],
    ...     ["this", "is", "the", "third", "document"],
    ... ]
    >>> lda_extractor = LdaTopicExtractor(corpus, num_topics=2)
    >>> topics = lda_extractor.extract_topics()
    >>> lda_extractor.print_topics(topics)
    >>> coherence, perplexity = lda_extractor.evaluate_lda()
    """

    def __init__(
        self,
        corpus: "MyCorpus",
        num_topics: int = 10,
        chunksize: int = 2_000,
        iterations: int = 400,
        passes: int = 30,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the TopicAnalyzer.

        Parameters
        ----------
        corpus : MyCorpus
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
        random_state : int, optional
            Random state for reproducibility, by default 42.
        """
        self.corpus: "MyCorpus" = corpus
        self.num_topics: int = num_topics
        self.chunksize: int = chunksize
        self.iterations: int = iterations
        self.passes: int = passes
        self.random_state: int = random_state
        self.lda_model: LdaModel | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_corpus={len(self.corpus):,})"  # type: ignore

    def _create_bigrams(self, min_count: int = 20, threshold: float = 2.0) -> Phrases:
        """
        Create bigrams from the corpus.

        Parameters
        ----------
        min_count : int, optional
            The minimum count of a bigram to be considered, by default 20.
        threshold : float, optional
            The threshold for considering a bigram, by default 2.0.

        Returns
        -------
        Phrases
            A Phrases model for creating bigrams.
        """
        return Phrases(sentences=self.corpus, min_count=min_count, threshold=threshold)

    def add_bigrams(self) -> "MyCorpus":  # type: ignore
        """
        Add bigrams to the corpus.

        Returns
        -------
        MyCorpus
            The corpus with added bigrams.
        """
        bigram: Phrases = self._create_bigrams()

        try:
            # Reset the iterator
            self.corpus.index = 0
        except AttributeError:
            console.print("Failed to reset iterator index")
            pass

        for doc in self.corpus:
            combined = doc + [token for token in bigram[doc] if "_" in token]
            yield combined

    @staticmethod
    def _clean_up_corpus(dictionary: Dictionary) -> None:
        """
        Clean up the corpus by filtering extreme tokens and compactifying the dictionary.

        Parameters
        ----------
        dictionary : Dictionary
            The gensim Dictionary object to be cleaned up.

        Returns
        -------
        None
        """
        min_token_freq: int = 2  # Keep tokens which are contained in at least no_below documents.
        # Keep tokens which are contained in no MORE than no_above documents.
        # (fraction of total number of documents)
        max_token_freq: float = 0.45

        # Filter low-freq and high-freq words
        dictionary.filter_extremes(no_below=min_token_freq, no_above=max_token_freq)

        # Remove gaps in id sequence after words are filtered
        dictionary.compactify()

    def create_bow(self) -> tuple[Dictionary, list[list[tuple[int, int]]]]:
        """
        Create a Bag of Words (BoW) representation of the corpus.

        Returns
        -------
        tuple[Dictionary, list[list[tuple[int, int]]]]
            A tuple containing the dictionary and the BoW corpus.
            The dictionary is a gensim Dictionary object.
            The BoW corpus is a list of documents, where each document is a list of
            (token_id, token_count) tuples.

        Notes
        -----
        The shape of the returned BoW corpus is (n_documents, n_unique_tokens),
        where n_documents is the number of documents in the corpus,
        and n_unique_tokens is the number of unique tokens in the dictionary.
        """
        self.corpus_wth_bigrams_: list[list[str]] = list(self.add_bigrams())
        dictionary_: Dictionary = Dictionary(self.corpus_wth_bigrams_)
        self._clean_up_corpus(dictionary_)
        corpus_bow: list[list[tuple[int, int]]] = [
            dictionary_.doc2bow(doc) for doc in self.corpus_wth_bigrams_
        ]
        return dictionary_, corpus_bow

    def train_model(self) -> LdaModel:
        """
        Train the LDA model.

        Returns
        -------
        LdaModel
            The trained LDA model.
        """
        dictionary_: Dictionary
        corpus_bow: list[list[tuple[int, int]]]
        dictionary_, corpus_bow = self.create_bow()
        try:
            self.lda_model = LdaModel(
                corpus=corpus_bow,
                id2word=dictionary_,
                chunksize=self.chunksize,
                alpha="auto",
                eta="auto",
                iterations=self.iterations,
                num_topics=self.num_topics,
                passes=self.passes,
                eval_every=None,
                random_state=self.random_state,
            )
            console.print("LDA model trained.", style="info")
        except Exception as e:
            console.print(e)

        return self.lda_model

    def extract_topics(self) -> list[tuple[list[tuple[str, float]], float]]:
        """
        Extract topics from the trained LDA model.

        Returns
        -------
        list[tuple[list[tuple[str, float]], float]]
            A list of tuples containing the top topics and their coherence scores.
        """
        topn: int = 10
        if self.lda_model is None:
            self.train_model()

        corpus_bow: list[list[tuple[int, int]]]
        _, corpus_bow = self.create_bow()
        self.top_topics: list[tuple[list[tuple[str, float]], float]] = self.lda_model.top_topics(  # type: ignore
            corpus=corpus_bow, topn=topn
        )
        avg_topic_coherence: float = sum([t[1] for t in self.top_topics]) / self.num_topics
        console.print(f"Average topic coherence: {avg_topic_coherence:.4f}\n", style="info")
        return self.top_topics

    @staticmethod
    def print_topics(top_topics: list[tuple[list[tuple[str, float]], float]]) -> None:
        """
        Print the top topics.

        Parameters
        ----------
        top_topics : list[tuple[list[tuple[str, float]], float]]
            A list of tuples containing the top topics and their coherence scores.
        """
        console.print(top_topics, style="info")

    @staticmethod
    def _compute_topic_coherence(
        model: LdaModel, corpus_with_bigrams: list[list[str]], dictionary: Dictionary
    ) -> float:
        """
        Compute the topic coherence score. Higher is better.

        Parameters
        ----------
        model : LdaModel
            The trained LDA model.
        corpus_with_bigrams : list[list[str]]
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
    def _compute_perplexity(model: LdaModel, corpus_bow: list[list[tuple[int, int]]]) -> float:
        """
        Compute the perplexity of the model. Lower is better

        Parameters
        ----------
        model : LdaModel
            The trained LDA model.
        corpus_bow : list[list[tuple[int, int]]]
            The BoW representation of the corpus.

        Returns
        -------
        float
            The computed perplexity.
        """
        perplexity: float = model.log_perplexity(corpus_bow)
        return perplexity

    def evaluate_lda(self) -> tuple[float, float]:
        """
        Evaluate the LDA model by computing coherence score and perplexity.

        Returns
        -------
        tuple[float, float]
            A tuple containing the coherence score and perplexity.
        """
        if self.lda_model is None:
            self.train_model()

        dictionary_: Dictionary
        corpus_bow: list[list[tuple[int, int]]]
        dictionary_, corpus_bow = self.create_bow()
        coherence_score: float = self._compute_topic_coherence(
            self.lda_model, self.corpus_wth_bigrams_, dictionary_
        )
        perplexity: float = self._compute_perplexity(self.lda_model, corpus_bow)
        console.print(
            f"N_topics: {self.num_topics} | Coherence score: {coherence_score:.4f} "
            f"| Perplexity: {perplexity:.4f}",
            style="info",
        )
        return coherence_score, perplexity
