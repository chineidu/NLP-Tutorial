"""This module is used to build a Text Classifier using Markov model."""

import numpy as np


class MarkovClassifier:
    """This classifier uses MArkov model for classifiication.\n
    It's trained using two models. i.e for 2 classes (labels).

    Params:
        word2idx (dict): A dictionary containing the vocabulary. i.e Bag of Words.

    Returns:
        None
    """

    def __init__(self, word2idx: dict) -> None:
        self.log_A_s = None
        self.log_Pi_s = None
        self.log_priors = None
        self.K = None
        self.word2idx = word2idx

    def __repr__(self) -> str:
        return f"{__class__.__name__}()"

    def _initialize_A_n_Pi(self) -> tuple[np.ndarray, np.ndarray]:
        """This is used to initialise the matrices."""
        # Since we have 2 classes (edgar_allan_poe, robert_frost),
        # we need to build 2 models. i.e A_0, Pi_0 and A_1, Pi_1
        V = len(self.word2idx)  # Vocabulary or number of states

        A = np.ones((V, V))  # Matrix with add-one smoothering
        Pi = np.ones(V)  # Vector with add-one smoothering
        return (A, Pi)

    def _compute_counts(
        self, X: list[int], A: np.ndarray, Pi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """This is used to populate A and Pi, normalize the values and
        calculate log probabilities. i.e count the number of occurrences
        of each state and divide by the total number of posible occurrences
        and find the log probability to prevent overflow error.

        Params:
            X: The vectorized doc. i.e List of int.
            A: The state transition matrix.
            Pi: The initial state distibution.

        Returns:
            (log_A, log_Pi): Tuple containing the log prob of A and Pi.
        """
        for tokenized_doc in X:
            prev_idx = None  # previous state/first word
            for idx in tokenized_doc:
                # If there's no prev state/word. i.e it's the first word.
                if prev_idx is None:
                    Pi[idx] += 1
                else:
                    # A prev word exists, count a transition
                    A[prev_idx, idx] += 1

                # Update the prev idx with the current idx
                prev_idx = idx
        return (A, Pi)

    def _calculate_probabilities(
        self, A: np.ndarray, Pi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """This is used to normalize the arrays. i.e calculate
        log probabilities."""
        A /= A.sum(axis=1, keepdims=True)  # Returns in 2-D array
        Pi /= Pi.sum()  # Returns in 1-D array
        return (np.log(A), np.log(Pi))

    def _compute_log_priors(self, y: np.ndarray) -> tuple[float, float]:
        count_0 = sum(_y == 0 for _y in y)
        count_1 = sum(_y == 1 for _y in y)
        total = len(y)
        p_0 = count_0 / total
        p_1 = count_1 / total
        log_p_0, log_p_1 = np.log(p_0), np.log(p_1)
        return (log_p_0, log_p_1)

    def _calculate_log_probs_n_priors(self, X: np.ndarray, y: np.ndarray) -> None:
        """This calculates the log probabilities for the two (2) models (classes) and log priors.
        It returns (log_A_0, log_A_1), (log_Pi_0, and log_Pi_1) and (log_p_0, log_p_1)."""
        # 1st model
        A_0, Pi_0 = self._initialize_A_n_Pi()
        A_0, Pi_0 = self._compute_counts(
            X=[_data for _data, _y in zip(X, y) if _y == 0],
            A=A_0,
            Pi=Pi_0,
        )
        log_A_0, log_Pi_0 = self._calculate_probabilities(A_0, Pi_0)

        # 2nd model
        A_1, Pi_1 = self._initialize_A_n_Pi()
        A_1, Pi_1 = self._compute_counts(
            X=[_data for _data, _y in zip(X, y) if _y == 1],
            A=A_1,
            Pi=Pi_1,
        )
        log_A_1, log_Pi_1 = self._calculate_probabilities(A_1, Pi_1)
        log_p_0, log_p_1 = self._compute_log_priors(y)

        self.log_A_s = [log_A_0, log_A_1]
        self.log_Pi_s = [log_Pi_0, log_Pi_1]
        self.log_priors = [log_p_0, log_p_1]
        self.K = len(self.log_priors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._calculate_log_probs_n_priors(X, y)
        return self

    def _compute_log_likelihood(self, input_: list[int], class_: int) -> None:
        """This returns the log of the probabilities."""
        # Extract the log of A and Pi for the given class (label)
        log_A = self.log_A_s[class_]
        log_Pi = self.log_Pi_s[class_]

        # Initialize variables
        prev_idx = None
        log_prob = 0

        for idx in input_:
            # If it's the first token, replace
            # it with the probability from log_Pi.
            if prev_idx is None:
                log_prob += log_Pi[idx]

            # replace it with the probability from log_A.
            else:
                log_prob += log_A[prev_idx, idx]

            # Update the value (for the next iteration)
            prev_idx = idx
        return log_prob

    def predict(self, X: list[list[int]]) -> list[int]:
        """This is used to make predictions using the trained Markov model."""
        # Initialize
        predictions = np.zeros(len(X))
        # Make predictions for every sentence in X
        for idx, input_ in enumerate(X):
            posteriors = [
                self._compute_log_likelihood(input_=input_, class_=class_) + self.log_priors[class_]
                for class_ in range(self.K)
            ]
            pred = np.argmax(posteriors)
            predictions[idx] = pred
        return predictions
