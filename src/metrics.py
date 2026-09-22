"""Metric utilities for classification training and evaluation."""

import torch

# Stage 5

class ClassificationMetrics:
    """Accumulate sample-weighted loss and classification accuracy."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0

    @torch.no_grad()    # 表示执行时，PyTorch 不需要为其中的运算建立反向传播计算图。
    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Add the statistics of one batch."""

        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)

        # loss 是当前 batch 的平均 loss，乘以样本数还原为总 loss。
        self.total_loss += loss.item() * batch_size

        # 比较预测类别和真实类别，累计预测正确的样本数。
        self.total_correct += (
            predictions == labels
        ).sum().item()

        self.total_samples += batch_size

    def compute(self) -> dict[str, float]:
        """Return the average loss and accuracy."""

        if self.total_samples == 0:
            raise ValueError(
                "Cannot compute metrics before any samples are added."
            )

        return {
            "loss": self.total_loss / self.total_samples,
            "accuracy": self.total_correct / self.total_samples,
        }