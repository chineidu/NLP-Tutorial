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
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

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
            Mask tensor of shape (batch_size, seq_len, seq_len).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Output tensor of shape (batch_size, seq_len, hidden_dim)
            - Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size: int = query.shape[0]

        Q: torch.Tensor = self.query(query)
        K: torch.Tensor = self.key(key)
        V: torch.Tensor = self.value(value)

        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy: torch.Tensor = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim**0.5

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention: torch.Tensor = torch.softmax(energy, dim=-1)

        out: torch.Tensor = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, -1, self.num_heads * self.head_dim)
        out = self.fc_out(out)

        return out, attention


class EnhancedLSTMClassifier(nn.Module):
    """
    Enhanced LSTM Classifier with Multi-Head Attention.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the embedding layer.
    hidden_dim : int
        Dimension of the hidden layer.
    output_dim : int
        Dimension of the output layer.
    n_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    num_heads : int, optional
        Number of attention heads (default is 4).
    num_tokens : int, optional
        Number of tokens (default is 15).
    max_seq_length : int, optional
        Maximum sequence length (default is 100).
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
        max_seq_length: int = 100,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim * num_tokens + 2,
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
        Forward pass of the Enhanced LSTM Classifier.

        Parameters
        ----------
        dates : torch.Tensor
            Tensor of dates with shape (batch_size, seq_length).
        input_ids : torch.Tensor
            Tensor of input ids with shape (batch_size, seq_length, num_tokens).
        amounts : torch.Tensor
            Tensor of amounts with shape (batch_size, seq_length).
        label : torch.Tensor | None, optional
            Tensor of labels with shape (batch_size,).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If label is provided:
                tuple containing (loss, logits)
            If label is None:
                logits
        """
        batch_size, seq_length, num_tokens = input_ids.shape

        positions: torch.Tensor = (
            torch.arange(0, seq_length).expand(batch_size, seq_length).to(input_ids.device)
        )

        token_embed: torch.Tensor = self.embedding(input_ids)
        pos_embed: torch.Tensor = self.pos_embedding(positions).unsqueeze(2)

        embedded: torch.Tensor = token_embed + pos_embed
        embedded = embedded.view(batch_size, seq_length, -1)

        dates = dates.unsqueeze(-1)
        amounts = amounts.unsqueeze(-1)

        combined: torch.Tensor = torch.cat((embedded, dates, amounts), dim=-1)

        lstm_output: torch.Tensor
        lstm_output, _ = self.lstm(combined)

        attention_output: torch.Tensor
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        attention_output = self.dropout(attention_output)

        out: torch.Tensor = self.layer_norm1(lstm_output + attention_output)

        ff_output: torch.Tensor = self.feed_forward(out)
        ff_output = self.dropout(ff_output)

        out = self.layer_norm2(out + ff_output)

        pooled_output: torch.Tensor = torch.max(out, dim=1)[0]

        logits: torch.Tensor = self.fc_layer(pooled_output)

        if label is not None:
            loss: torch.Tensor = F.cross_entropy(logits, label)
            return (loss, logits)
        return logits
