from typing import Any

from torch import Tensor, nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
)


class Transformation:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        padding: bool = True,
        truncation: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(self, texts: str | list[str]) -> dict[str, Any]:
        encoded_input: dict[str, Any] = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded_input

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tokenizer})"


class BackBone(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str, fine_tune: bool = True) -> None:
        super().__init__()

        self.model_path = pretrained_model_name_or_path
        self.backbone = self._get_backbone()
        if fine_tune:
            self.prepare_for_fine_tuning()

        self.total_params = self._get_total_params()
        self.trainable_params = self._get_trainable_params()

    def forward(self, enc_input: Tensor) -> Tensor:
        output: Tensor = self.backbone(**enc_input)
        # Extract the [CLS] token's hidden state, which represents the entire input
        # sequence in bidirectional models like BERT/DistilBERT.
        cls_output: Tensor = output.last_hidden_state[:, 0, :]
        return cls_output

    def _get_backbone(self) -> AutoModel:
        return AutoModel.from_pretrained(self.model_path)

    def _get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def prepare_for_fine_tuning(self) -> None:
        """
        Prepares the model for fine-tuning by freezing most of the model's parameters.
        This allows the model to be fine-tuned on a specific task while preserving the
        learned representations from the pre-trained model.
        """
        # Get the model ready for fine-tuning by `freezing` the model's parameters.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Make the last `transformer block` layer trainable
        for param in self.backbone.transformer.layer[-1].parameters():
            param.requires_grad = True

        # Make the last `output_layer_norm` layer trainable
        for param in self.backbone.transformer.layer[-1].output_layer_norm.parameters():
            param.requires_grad = True


class SoftmaxHead(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, dim: int = -1) -> None:
        super().__init__()

        self.head = nn.Sequential(nn.Linear(emb_dim, num_classes), nn.Softmax(dim=dim))

    def forward(self, x: Tensor) -> Tensor:
        probas: Tensor = self.head(x)
        return probas


class SpendClassifier(nn.Module):
    def __init__(self, backbone: BackBone, head: SoftmaxHead) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, enc_input: BatchEncoding) -> Tensor:
        cls_output: Tensor = self.backbone(enc_input)
        probas: Tensor = self.head(cls_output).squeeze(0)
        return probas
