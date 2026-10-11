"""Tests for Stage 16 MiniViT transfer-learning semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.minivit import MiniViT
from src.transfer import (
    TransferMiniViT,
    build_transfer_model,
    clone_encoder_state,
    encoder_state_is_unchanged,
    trainable_parameter_names,
    trainable_parameters,
)


MODEL_CONFIG = {
    "name": "minivit",
    "image_size": 8,
    "in_channels": 3,
    "num_classes": 10,
    "patch_size": 4,
    "embed_dim": 8,
    "depth": 1,
    "num_heads": 2,
    "mlp_dim": 16,
    "dropout": 0.2,
}


def _write_source_checkpoint(
    path: Path,
    *,
    model_config: dict[str, object] | None = None,
) -> MiniViT:
    config = dict(MODEL_CONFIG if model_config is None else model_config)
    source = MiniViT(**{
        key: value
        for key, value in config.items()
        if key != "name"
    })
    # 使用极端常数，使“旧 head 被错误加载”必然可以被测试发现。
    with torch.no_grad():
        source.classifier.weight.fill_(17.0)
        source.classifier.bias.fill_(-9.0)
    torch.save(
        {
            "model_state": source.state_dict(),
            "config": {"model": config},
        },
        path,
    )
    return source


def _one_update(model: TransferMiniViT) -> None:
    optimizer = torch.optim.SGD(
        trainable_parameters(model),
        lr=0.1,
    )
    model.train()
    logits = model(torch.randn(4, 3, 8, 8))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3]))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_minivit_forward_features_separates_encoder_and_head() -> None:
    model = MiniViT(**{
        key: value
        for key, value in MODEL_CONFIG.items()
        if key != "name"
    })
    # 两次独立 forward 只有在 Dropout 关闭时才应逐元素等价。
    model.eval()
    images = torch.randn(2, 3, 8, 8)

    features = model.forward_features(images)
    logits = model(images)

    assert features.shape == (2, 8)
    assert logits.shape == (2, 10)
    assert torch.allclose(logits, model.classifier(features))


def test_scratch_is_random_and_all_parameters_are_trainable() -> None:
    model, initialization = build_transfer_model(
        MODEL_CONFIG,
        mode="scratch",
        device=torch.device("cpu"),
    )

    assert initialization.source_checkpoint is None
    assert initialization.loaded_encoder_key_count == 0
    assert set(trainable_parameter_names(model)) == {
        name for name, _ in model.named_parameters()
    }
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 10)


@pytest.mark.parametrize("mode", ["linear_probe", "fine_tune"])
def test_pretrained_modes_load_encoder_but_discard_source_head(
    tmp_path: Path,
    mode: str,
) -> None:
    checkpoint = tmp_path / "best.pt"
    source = _write_source_checkpoint(checkpoint)

    model, initialization = build_transfer_model(
        MODEL_CONFIG,
        mode=mode,
        device=torch.device("cpu"),
        source_checkpoint=checkpoint,
    )

    source_state = source.state_dict()
    for name, value in model.state_dict().items():
        if not name.startswith("classifier."):
            assert torch.equal(value, source_state[name])
    assert not torch.equal(
        model.classifier.weight,
        source.classifier.weight,
    )
    assert not torch.equal(model.classifier.bias, source.classifier.bias)
    assert initialization.discarded_source_keys == (
        "classifier.bias",
        "classifier.weight",
    )
    assert initialization.source_checkpoint_sha256 is not None


def test_linear_probe_exposes_only_classifier_to_optimizer(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    _write_source_checkpoint(checkpoint)
    model, _ = build_transfer_model(
        MODEL_CONFIG,
        mode="linear_probe",
        device=torch.device("cpu"),
        source_checkpoint=checkpoint,
    )

    assert trainable_parameter_names(model) == (
        "classifier.weight",
        "classifier.bias",
    )
    assert sum(
        parameter.numel()
        for parameter in trainable_parameters(model)
    ) == 8 * 10 + 10
    assert all(
        not parameter.requires_grad
        for _, parameter in model.encoder_named_parameters()
    )


def test_linear_probe_train_mode_keeps_encoder_eval_and_head_train(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    _write_source_checkpoint(checkpoint)
    model, _ = build_transfer_model(
        MODEL_CONFIG,
        mode="linear_probe",
        device=torch.device("cpu"),
        source_checkpoint=checkpoint,
    )

    model.train()

    assert model.training is True
    assert model.patch_embedding.training is False
    assert model.blocks.training is False
    assert model.final_norm.training is False
    assert model.classifier.training is True


def test_linear_probe_update_changes_head_but_not_encoder(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    _write_source_checkpoint(checkpoint)
    model, _ = build_transfer_model(
        MODEL_CONFIG,
        mode="linear_probe",
        device=torch.device("cpu"),
        source_checkpoint=checkpoint,
    )
    encoder_before = clone_encoder_state(model)
    head_before = model.classifier.weight.detach().clone()

    _one_update(model)

    assert encoder_state_is_unchanged(model, encoder_before)
    assert not torch.equal(model.classifier.weight, head_before)
    assert all(
        parameter.grad is None
        for _, parameter in model.encoder_named_parameters()
    )


def test_fine_tune_update_changes_encoder_and_head(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    _write_source_checkpoint(checkpoint)
    model, _ = build_transfer_model(
        MODEL_CONFIG,
        mode="fine_tune",
        device=torch.device("cpu"),
        source_checkpoint=checkpoint,
    )
    encoder_before = clone_encoder_state(model)
    head_before = model.classifier.weight.detach().clone()

    _one_update(model)

    assert not encoder_state_is_unchanged(model, encoder_before)
    assert not torch.equal(model.classifier.weight, head_before)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_linear_probe_attention_output_remains_available(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    _write_source_checkpoint(checkpoint)
    model, _ = build_transfer_model(
        MODEL_CONFIG,
        mode="linear_probe",
        device=torch.device("cpu"),
        source_checkpoint=checkpoint,
    )

    logits, attention = model(
        torch.randn(2, 3, 8, 8),
        return_attention=True,
    )

    assert logits.shape == (2, 10)
    assert attention.shape == (2, 2, 5, 5)


def test_linear_probe_all_attention_layers_preserve_freeze():
    """Linear Probe 收集全部层时 encoder 仍冻结，分类头仍可反传。"""

    model = TransferMiniViT(
        transfer_mode="linear_probe",
        image_size=8,
        patch_size=4,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_dim=16,
        dropout=0.2,
    )
    model.train()
    images = torch.randn(2, 3, 8, 8)

    logits, attention = model(
        images,
        return_attention=True,
        attention_layers="all",
    )
    logits.sum().backward()

    assert attention.shape == (2, 2, 2, 5, 5)
    assert not attention.requires_grad
    assert model.classifier.weight.grad is not None
    assert all(
        parameter.grad is None
        for _, parameter in model.encoder_named_parameters()
    )
    assert not model.blocks.training


def test_pretrained_mode_rejects_encoder_architecture_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    source_config = {**MODEL_CONFIG, "patch_size": 2}
    _write_source_checkpoint(checkpoint, model_config=source_config)

    with pytest.raises(ValueError, match="architectures differ"):
        build_transfer_model(
            MODEL_CONFIG,
            mode="fine_tune",
            device=torch.device("cpu"),
            source_checkpoint=checkpoint,
        )


def test_pretrained_mode_rejects_incomplete_encoder_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    _write_source_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    del payload["model_state"]["cls_token"]
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="does not exactly match"):
        build_transfer_model(
            MODEL_CONFIG,
            mode="linear_probe",
            device=torch.device("cpu"),
            source_checkpoint=checkpoint,
        )


def test_checkpoint_presence_contract_is_explicit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    _write_source_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="must not"):
        build_transfer_model(
            MODEL_CONFIG,
            mode="scratch",
            device=torch.device("cpu"),
            source_checkpoint=checkpoint,
        )
    with pytest.raises(ValueError, match="requires"):
        build_transfer_model(
            MODEL_CONFIG,
            mode="linear_probe",
            device=torch.device("cpu"),
        )
