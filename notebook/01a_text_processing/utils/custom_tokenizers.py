import os
from collections import Counter
from typing import List

from transformers import PreTrainedTokenizerFast

from utils import tokenize_by_special_chars  # type: ignore

# Set the environment variable to disable the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    def __init__(self, max_vocab_size: int = 70_000):
        """
        Initialize the RegexTokenizer.

        Args:
            max_vocab_size (int): Maximum size of the vocabulary. Defaults to 70,000.
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


class WordPieceTokenizer:
    """
    A custom tokenizer that uses WordPiece to tokenize text and build a vocabulary.

    Attributes
    ----------
    UNK : int
        Unknown token index.
    PAD : int
        Padding token index.
    tokenizer : PreTrainedTokenizerFast
        The underlying WordPiece tokenizer.
    vocab_size : int
        Size of the vocabulary.
    """

    UNK: int = 0
    PAD: int = 1

    def __init__(self, tokenizer_file: str) -> None:
        """
        Initialize the WordPieceTokenizer.

        Parameters
        ----------
        tokenizer_file : str
            Path to the tokenizer file.
        """
        self.tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        self.vocab_size: int = self.tokenizer.vocab_size

    def __repr__(self) -> str:
        """
        Return a string representation of the WordPieceTokenizer.

        Returns
        -------
        str
            A string representation of the WordPieceTokenizer.
        """
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size:,})"

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.

        Parameters
        ----------
        text : str
            The input text to tokenize.

        Returns
        -------
        List[str]
            A list of tokens.
        """
        tokens: List[str] = self.tokenizer.tokenize(text)
        return tokens

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of input texts.

        Parameters
        ----------
        texts : List[str]
            A list of input texts to tokenize.

        Returns
        -------
        List[List[str]]
            A list of lists of tokens, where each inner list corresponds to a tokenized
            input text.
        """
        return [self.tokenize(text) for text in texts]

    def encode(self, text: str, max_length: int = 10) -> List[int]:
        """
        Encode the input text into a list of token IDs.

        Parameters
        ----------
        text : str
            The input text to encode.
        max_length : int, optional
            The maximum length of the encoded sequence, by default 10.

        Returns
        -------
        List[int]
            A list of token IDs.
        """
        tokens: List[str] = self.tokenizer.tokenize(text)
        encoded: List[int] = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(encoded) < max_length:
            encoded += [self.PAD] * (max_length - len(encoded))
        else:
            encoded = encoded[:max_length]
        return encoded

    def batch_encode(self, texts: List[str], max_length: int = 10) -> List[List[int]]:
        """
        Encode a batch of input texts into lists of token IDs.

        Parameters
        ----------
        texts : List[str]
            A list of input texts to encode.
        max_length : int, optional
            The maximum length of each encoded sequence, by default 10.

        Returns
        -------
        List[List[int]]
            A list of lists of token IDs, where each inner list corresponds to
            an encoded input text.
        """
        return [self.encode(text, max_length) for text in texts]

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Parameters
        ----------
        ids : List[int]
            A list of token IDs to decode.

        Returns
        -------
        str
            The decoded text.
        """
        return self.tokenizer.decode(ids)
