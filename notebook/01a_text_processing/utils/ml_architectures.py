import torch
from torch import nn
from torch.functional import F


# Simple
class SimpleLSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequence data.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the embedding layer.
    hidden_dim : int
        Dimension of the hidden state in LSTM.
    output_dim : int
        Dimension of the output (number of classes).
    n_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.

    Attributes
    ----------
    embedding : nn.Embedding
        Embedding layer.
    lstm : nn.LSTM
        LSTM layer.
    dropout : nn.Dropout
        Dropout layer.
    fc_layer : nn.Sequential
        Fully connected layers for classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim + 2,  # +2 for date and amount
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        dates: torch.Tensor,
        input_ids: torch.Tensor,
        amounts: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTM classifier.

        Parameters
        ----------
        dates : torch.Tensor
            Tensor of shape (batch_size, max_transactions) containing date information.
        input_ids : torch.Tensor
            Tensor of shape (batch_size, max_transactions, max_length) containing
            input token ids.
        amounts : torch.Tensor
            Tensor of shape (batch_size, max_transactions) containing amount information.
        label : torch.Tensor | None, optional
            Tensor of shape (batch_size,) containing true labels, by default None.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If label is provided (training mode):
                tuple containing (loss: torch.Tensor, logits: torch.Tensor)
            If label is None (inference mode):
                logits: torch.Tensor of shape (batch_size, output_dim)
        """
        # [batch_size, max_transactions, max_length, embedding_dim]
        out: torch.Tensor = self.embedding(input_ids)

        # Reshape dates and amounts. (Add a new dimension at the end)
        dates = dates.unsqueeze(-1)  # [batch_size, max_transactions, 1]
        amounts = amounts.unsqueeze(-1)  # [batch_size, max_transactions, 1]

        # Concatenate along the last dimension
        # combined shape: [batch_size, max_transactions, embedding_dim + 2]
        combined = torch.cat((out.sum(dim=2), dates, amounts), dim=-1)

        hidden: torch.Tensor
        _, (hidden, _) = self.lstm(combined)

        # Use the last layer (forward and backward if bidirectional)
        if self.lstm.bidirectional:
            out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            out = hidden[-1, :, :]

        out = self.dropout(out)
        logits: torch.Tensor = self.fc_layer(out)

        if label is not None:
            # Training mode
            loss: torch.Tensor = F.cross_entropy(logits, label)
            return (loss, logits)
        # Inference mode
        return logits


# Simple Attention
class AttentionLayer(nn.Module):
    """
    Attention layer for LSTM output.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden state.

    Attributes
    ----------
    attention : nn.Linear
        Linear layer for computing attention weights.

    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.

        Parameters
        ----------
        lstm_output : torch.Tensor
            Output from LSTM layer, shape (batch_size, seq_len, hidden_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            context_vector : torch.Tensor
                Weighted sum of lstm_output, shape (batch_size, hidden_dim).
            attention_weights : torch.Tensor
                Attention weights, shape (batch_size, seq_len, 1).
        """
        attention_weights: torch.Tensor = F.softmax(self.attention(lstm_output), dim=1)
        context_vector: torch.Tensor = torch.sum(attention_weights * lstm_output, dim=1)

        return context_vector, attention_weights


class LSTMClassifier(nn.Module):
    """
    LSTM Classifier with attention mechanism.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the embedding layer.
    hidden_dim : int
        Dimension of the hidden state in LSTM.
    output_dim : int
        Dimension of the output (number of classes).
    n_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.

    Attributes
    ----------
    embedding : nn.Embedding
        Embedding layer.
    lstm : nn.LSTM
        LSTM layer.
    attention : AttentionLayer
        Attention layer.
    layer_norm : nn.LayerNorm
        Layer normalization.
    dropout : nn.Dropout
        Dropout layer.
    fc_layer : nn.Sequential
        Fully connected layers for classification.

    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim + 2,  # +2 for date and amount
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        dates: torch.Tensor,
        input_ids: torch.Tensor,
        amounts: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTM Classifier.

        Parameters
        ----------
        dates : torch.Tensor
            Tensor of dates, shape (batch_size, seq_len).
        input_ids : torch.Tensor
            Tensor of input ids, shape (batch_size, seq_len).
        amounts : torch.Tensor
            Tensor of amounts, shape (batch_size, seq_len).
        label : torch.Tensor | None, optional
            Tensor of labels, shape (batch_size,).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If label is provided:
                tuple containing (loss, logits)
            If label is None:
                logits
        """
        out: torch.Tensor = self.embedding(input_ids)

        # Reshape dates and amounts to match the shape of the embedding output
        dates = dates.unsqueeze(-1)
        amounts = amounts.unsqueeze(-1)

        combined: torch.Tensor = torch.cat((out.sum(dim=2), dates, amounts), dim=-1)

        lstm_output: torch.Tensor
        lstm_output, _ = self.lstm(combined)

        # Apply attention
        context_vector: torch.Tensor
        context_vector, _ = self.attention(lstm_output)

        # Apply layer normalization
        normalized_output: torch.Tensor = self.layer_norm(context_vector)

        out = self.dropout(normalized_output)
        logits: torch.Tensor = self.fc_layer(out)

        if label is not None:
            loss: torch.Tensor = F.cross_entropy(logits, label)
            return (loss, logits)
        return logits


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden layer.
    num_heads : int
        Number of attention heads.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    query : nn.Linear
        Linear layer for query projection.
    key : nn.Linear
        Linear layer for key projection.
    value : nn.Linear
        Linear layer for value projection.
    fc_out : nn.Linear
        Linear layer for output projection.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads

        self.query: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.key: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.value: nn.Linear = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out: nn.Linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Multi-Head Attention module.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len, hidden_dim).
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len, hidden_dim).
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len, hidden_dim).
        mask : torch.Tensor | None, optional
            Mask tensor of shape (batch_size, seq_len, seq_len), by default None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - out: Output tensor of shape (batch_size, seq_len, hidden_dim).
            - attn_weights: Attention weights tensor of shape
            (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size: int = query.shape[0]

        Q: torch.Tensor = self.query(query)
        K: torch.Tensor = self.key(key)
        V: torch.Tensor = self.value(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores: torch.Tensor = torch.matmul(Q, K.transpose(3, 2)) / self.head_dim**0.5

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights: torch.Tensor = torch.softmax(attn_scores, dim=-1)

        out: torch.Tensor = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.fc_out(out)

        return out, attn_weights


class EnhancedLSTMClassifier(nn.Module):
    """
    Enhanced LSTM Classifier with attention mechanism and positional embeddings.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the embedding vectors.
    hidden_dim : int
        Dimension of the hidden state in LSTM.
    output_dim : int
        Dimension of the output (number of classes).
    n_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    num_heads : int, optional
        Number of attention heads, by default 4.
    num_tokens : int, optional
        Number of tokens per transaction, by default 15.
    max_transactions : int, optional
        Maximum number of transactions, by default 100.

    Attributes
    ----------
    embedding : nn.Embedding
        Token embedding layer.
    pos_embedding : nn.Embedding
        Positional embedding layer.
    lstm : nn.LSTM
        LSTM layer.
    attention : MultiHeadAttention
        Multi-head attention layer.
    layer_norm1 : nn.LayerNorm
        Layer normalization 1.
    layer_norm2 : nn.LayerNorm
        Layer normalization 2.
    feed_forward : nn.Sequential
        Feed-forward neural network.
    dropout : nn.Dropout
        Dropout layer.
    fc_layer : nn.Sequential
        Fully connected output layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        num_heads: int = 4,
        num_tokens: int = 15,
        max_transactions: int = 100,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_transactions, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim * num_tokens + 7,  # Multiply by num_tokens, +7 for date and amount
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = MultiHeadAttention(hidden_dim * 2, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
        )

        self.dropout = nn.Dropout(dropout)

        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        dates: torch.Tensor,
        input_ids: torch.Tensor,
        amounts: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the EnhancedLSTMClassifier.

        Parameters
        ----------
        dates : torch.Tensor
            Tensor of shape (batch_size, seq_length, 6) containing date features.
        input_ids : torch.Tensor
            Tensor of shape (batch_size, seq_length, num_tokens) containing token IDs.
        amounts : torch.Tensor
            Tensor of shape (batch_size, seq_length) containing transaction amounts.
        label : torch.Tensor | None, optional
            Tensor of shape (batch_size,) containing labels, by default None.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If label is None:
                logits: Tensor of shape (batch_size, output_dim) containing class logits.
            If label is not None:
                tuple containing:
                - loss: Scalar tensor with the computed loss.
                - logits: Tensor of shape (batch_size, output_dim) containing class logits.
        """
        batch_size, seq_length, _ = input_ids.shape

        # Create position tensor
        positions: torch.Tensor = (
            torch.arange(0, seq_length).expand(batch_size, seq_length).to(input_ids.device)
        )

        # Get embeddings
        token_embed: torch.Tensor = self.embedding(
            input_ids
        )  # Shape: [batch_size, seq_length, num_tokens, embedding_dim]
        pos_embed: torch.Tensor = self.pos_embedding(positions).unsqueeze(
            2
        )  # Shape: [batch_size, seq_length, 1, embedding_dim]

        # Combine token and positional embeddings
        embedded: torch.Tensor = token_embed + pos_embed

        # Reshape embedded to [batch_size, seq_length, num_tokens * embedding_dim]
        embedded = embedded.view(batch_size, seq_length, -1)

        amounts = amounts.unsqueeze(-1)

        combined: torch.Tensor = torch.cat(
            (embedded, dates, amounts), dim=-1
        )  # Shape: [batch_size, seq_length, num_tokens * embedding_dim + 7]

        lstm_output: torch.Tensor
        lstm_output, _ = self.lstm(combined)  # Shape: [batch_size, seq_length, hidden_dim * 2]

        attention_output: torch.Tensor
        attention_output, _ = self.attention(
            lstm_output, lstm_output, lstm_output
        )  # Shape: [batch_size, seq_length, hidden_dim * 2]
        attention_output = self.dropout(
            attention_output
        )  # Shape: [batch_size, seq_length, hidden_dim * 2]

        out: torch.Tensor = self.layer_norm1(
            lstm_output + attention_output
        )  # Shape: [batch_size, seq_length, hidden_dim * 2]

        ff_output: torch.Tensor = self.feed_forward(
            out
        )  # Shape: [batch_size, seq_length, hidden_dim * 2]
        ff_output = self.dropout(ff_output)  # Shape: [batch_size, seq_length, hidden_dim * 2]

        out = self.layer_norm2(out + ff_output)  # Shape: [batch_size, seq_length, hidden_dim * 2]

        pooled_output: torch.Tensor = torch.max(out, dim=1)[
            0
        ]  # Shape: [batch_size, hidden_dim * 2]

        logits: torch.Tensor = self.fc_layer(pooled_output)  # Shape: [batch_size, output_dim]

        if label is not None:
            loss: torch.Tensor = F.cross_entropy(logits, label)
            return (loss, logits)  # Shape: (scalar, [batch_size, output_dim])
        return logits  # Shape: [batch_size, output_dim]


class ImprovedLSTMClassifier(nn.Module):
    """
    Improved LSTM Classifier with separate LSTMs for date, narration, and amount.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the embedding vectors.
    hidden_dim : int
        Dimension of the hidden state in LSTM layers.
    output_dim : int
        Dimension of the output (number of classes).
    n_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    num_heads : int, optionalImprovedLSTMClassifier
        Number of attention heads, by default 4.
    num_tokens : int, optional
        Number of tokens in each transaction, by default 15.
    max_transactions : int, optional
        Maximum number of transactions, by default 100.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        num_heads: int = 4,
        num_tokens: int = 15,
        max_transactions: int = 100,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_transactions, embedding_dim)

        # Separate LSTMs for date, narration, and amount
        self.date_lstm = nn.LSTM(
            6,  # 6 features for date
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.narration_lstm = nn.LSTM(
            embedding_dim * num_tokens,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.amount_lstm = nn.LSTM(
            1,  # 1 feature for amount
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Multiply by 6 because of date, narration, and amount LSTM bidirectional outputs.
        # Each bidirectional output has 2 hidden states, so multiply by 2.
        self.attention = MultiHeadAttention(hidden_dim * 6, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 6)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 6)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 12),
            nn.ReLU(),
            nn.Linear(hidden_dim * 12, hidden_dim * 6),
        )

        self.dropout = nn.Dropout(dropout)

        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        dates: torch.Tensor,
        input_ids: torch.Tensor,
        amounts: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ImprovedLSTMClassifier.

        Parameters
        ----------
        dates : torch.Tensor
            Tensor of shape (batch_size, seq_length, 6) containing date features.
        input_ids : torch.Tensor
            Tensor of shape (batch_size, seq_length, num_tokens) containing token IDs.
        amounts : torch.Tensor
            Tensor of shape (batch_size, seq_length) containing transaction amounts.
        label : torch.Tensor | None, optional
            Tensor of shape (batch_size,) containing labels, by default None.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If label is provided, returns a tuple of (loss, logits).
            Otherwise, returns logits.
            logits shape: (batch_size, output_dim)
        """
        batch_size, seq_length, _ = input_ids.shape

        # Process dates
        date_output: torch.Tensor
        date_output, _ = self.date_lstm(dates)  # Shape: (batch_size, seq_length, hidden_dim * 2)

        # Process narration
        positions: torch.Tensor = (
            torch.arange(0, seq_length).expand(batch_size, seq_length).to(input_ids.device)
        )
        token_embed: torch.Tensor = self.embedding(input_ids)
        pos_embed: torch.Tensor = self.pos_embedding(positions).unsqueeze(2)
        embedded: torch.Tensor = token_embed + pos_embed
        embedded = embedded.view(batch_size, seq_length, -1)
        narration_output: torch.Tensor
        narration_output, _ = self.narration_lstm(
            embedded
        )  # Shape: (batch_size, seq_length, hidden_dim * 2)

        # Process amounts
        amount_output: torch.Tensor
        amount_output, _ = self.amount_lstm(
            amounts.unsqueeze(-1)
        )  # Shape: (batch_size, seq_length, hidden_dim * 2)

        # Combine outputs with equal weighting
        combined_output: torch.Tensor = torch.cat(
            (date_output, narration_output, amount_output), dim=-1
        )  # Shape: (batch_size, seq_length, hidden_dim * 6)

        attention_output: torch.Tensor
        attention_output, _ = self.attention(
            combined_output, combined_output, combined_output
        )  # Shape: (batch_size, seq_length, hidden_dim * 6)
        attention_output = self.dropout(attention_output)

        out: torch.Tensor = self.layer_norm1(combined_output + attention_output)

        ff_output: torch.Tensor = self.feed_forward(out)
        ff_output = self.dropout(ff_output)

        out = self.layer_norm2(out + ff_output)

        pooled_output: torch.Tensor = torch.max(out, dim=1)[
            0
        ]  # Shape: (batch_size, hidden_dim * 6)

        logits: torch.Tensor = self.fc_layer(pooled_output)  # Shape: (batch_size, output_dim)

        if label is not None:
            loss: torch.Tensor = F.cross_entropy(logits, label)
            return (loss, logits)
        return logits
