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
        self.total_correct = 0.0
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

        if labels.ndim == 1:
            # 硬标签：每个样本的正确贡献只能是 0 或 1。
            correct = (predictions == labels).sum().item()
        elif labels.ndim == 2 and labels.shape == logits.shape:
            # 软标签：top-1 类别获得的目标概率质量就是该样本的贡献。
            correct = labels.gather(
                dim=1,
                index=predictions.unsqueeze(1),
            ).sum().item()
        else:
            raise ValueError(
                "labels must have shape [batch_size] or match logits shape."
            )

        self.total_correct += correct

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
