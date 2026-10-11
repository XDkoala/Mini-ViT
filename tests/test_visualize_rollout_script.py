"""Tests for reproducible validation-only attention case selection."""

from types import SimpleNamespace

import torch

import scripts.visualize_rollout as rollout_script


def test_case_selection_uses_first_distinct_labels_for_each_status(monkeypatch):
    """Select in validation order, with separate clean correct/wrong quotas."""

    # 重复类别不能挤掉后面类别；选择规则只能使用 clean 预测。
    labels = [0, 0, 1, 2, 1, 3]
    predicted = [0, 0, 0, 2, 1, 0]
    samples = [{"image": i, "true_label": label} for i, label in enumerate(labels)]
    runtime = SimpleNamespace(prepare_image=lambda index: (torch.full((3, 32, 32), float(index)), None))

    def fake_prediction(runtime, inputs):
        probabilities = torch.zeros(len(inputs), 10)
        for row, tensor in enumerate(inputs):
            probabilities[row, predicted[int(tensor[0, 0, 0])]] = 1
        return SimpleNamespace(probabilities=probabilities)

    monkeypatch.setattr(rollout_script, "predict_batch", fake_prediction)
    selected, scanned = rollout_script.select_cases(runtime, samples, count=2, batch_size=2)
    assert [sample["image"] for sample in selected["correct"]] == [0, 3]
    assert [sample["image"] for sample in selected["wrong"]] == [2, 5]
    assert scanned == 6


def test_cifar_diagnostics_opens_only_training_source(monkeypatch):
    """Read the registered validation indices without opening test images."""

    # train=True 是验证集的原始来源；禁止顺手构建 test dataset。
    names = tuple(str(i) for i in range(10))

    class FakeCIFAR:
        def __init__(self, root, *, train, download):
            assert train is True
            assert download is False
            self.classes = names

        def __len__(self):
            return 50_000

        def __getitem__(self, index):
            return f"image_{index}", index % 10

    monkeypatch.setattr(rollout_script, "CIFAR10", FakeCIFAR)
    monkeypatch.setattr(rollout_script, "split_train_val_indices", lambda *args: ([], [17, 25, 31]))
    runtime = SimpleNamespace(dataset="cifar10", class_names=names,
                              config={"data": {"data_dir": "data"}, "seed": 42})
    samples = rollout_script.validation_samples(runtime, 2)
    assert [sample["source_index"] for sample in samples] == [17, 25]
