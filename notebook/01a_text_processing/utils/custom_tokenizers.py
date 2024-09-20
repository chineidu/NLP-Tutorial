from collections import Counter

from utils import tokenize_by_special_chars  # type: ignore


class RegexTokenizer:
    """
    A custom tokenizer that uses regex to tokenize text and build a vocabulary.

    Attributes:
        PAD (int): Padding token index.
        UNK (int): Unknown token index.
        max_vocab_size (int): Maximum size of the vocabulary.
        word2idx (dict[str, int]): Mapping of words to indices.
        idx2word (dict[int, str]): Mapping of indices to words.
        word_freq (Counter): Counter for word frequencies.

    """

    PAD: int = 0
    UNK: int = 1

    def __init__(self, max_vocab_size: int = 100_000):
        """
        Initialize the RegexTokenizer.

        Args:
            max_vocab_size (int): Maximum size of the vocabulary. Defaults to 30,000.
        """
        self.max_vocab_size: int = max_vocab_size
        self.word2idx: dict[str, int] = {"<PAD>": self.PAD, "<UNK>": self.UNK}
        self.idx2word: dict[int, str] = {self.PAD: "<PAD>", self.UNK: "<UNK>"}
        self.word_freq: Counter = Counter()  # type: ignore

    def __repr__(self) -> str:
        """
        Return a string representation of the RegexTokenizer.

        Returns:
            str: A string representation of the RegexTokenizer.
        """
        return (
            f"{self.__class__.__name__}(max_vocab_size={self.max_vocab_size:,}; "
            f"vocab_size={self.vocab_size:,})"
        )

    def fit(self, texts: list[str]) -> None:
        """
        Fit the tokenizer on the given texts.

        Args:
            texts (list[str]): list of input texts.
        """
        words: list[str] = [word for text in texts for word in self.tokenize(text)]
        # Add tokens
        self.word_freq.update(words)
        for word, _ in self.word_freq.most_common(self.max_vocab_size - 2):
            idx: int = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def tokenize(self, text: str | list[str]) -> list[str]:
        """
        Tokenize the input text.

        Args:
            text (str|list[str]): Input text to tokenize.

        Returns:
            list[str]: list of tokens.
        """
        tokens: list[str] = tokenize_by_special_chars(text, drop_digits=True)
        return tokens

    def encode(self, text: str, max_length: int) -> list[int]:
        """
        Encode the input text into a list of token indices.

        Args:
            text (str): Input text to encode.
            max_length (int): Maximum length of the encoded sequence.

        Returns:
            list[int]: list of token indices.
        """
        tokens: list[str] = self.tokenize(text)
        encoded: list[int] = [self.word2idx.get(token, self.UNK) for token in tokens]
        if len(encoded) < max_length:
            encoded += [self.PAD] * (max_length - len(encoded))
        else:
            encoded = encoded[:max_length]
        return encoded

    def batch_encode(self, texts: list[str], max_length: int) -> list[list[int]]:
        """
        Encode a batch of texts into lists of token indices.

        Args:
            texts (list[str]): list of input texts to encode.
            max_length (int): Maximum length of each encoded sequence.

        Returns:
            list[list[int]]: list of lists of token indices.
        """
        return [self.encode(text, max_length) for text in texts]

    @property
    def vocab_size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            int: Size of the vocabulary.
        """
        return len(self.word2idx)
