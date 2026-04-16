from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

INPUT_DIM = 124
TARGET_DIM = 128
LEVELS = 60
PROFILE_VARS = 2
SCALAR_VARS = 4
CHANNELS = PROFILE_VARS + SCALAR_VARS
FLUX_START = 120

# Order of scalar outputs in the 128-target subset:
# NETSW, FLWDS, PRECSC, PRECC, SOLS, SOLL, SOLSD, SOLLD
IDX_NETSW = FLUX_START + 0
IDX_FLWDS = FLUX_START + 1
IDX_PRECSC = FLUX_START + 2
IDX_PRECC = FLUX_START + 3
IDX_SOLS = FLUX_START + 4
IDX_SOLL = FLUX_START + 5
IDX_SOLSD = FLUX_START + 6
IDX_SOLLD = FLUX_START + 7


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, array: np.ndarray) -> "Standardizer":
        mean = np.asarray(array.mean(axis=0), dtype=np.float32)
        std = np.asarray(array.std(axis=0), dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, array: np.ndarray) -> np.ndarray:
        return (np.asarray(array, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float32) * self.std + self.mean


@dataclass
class PreparedData:
    x_train_flat: np.ndarray
    y_train_cls: np.ndarray
    x_val_flat: np.ndarray
    y_val_cls: np.ndarray
    x_eval_flat: np.ndarray
    y_eval_cls: np.ndarray
    x_train_seq: np.ndarray
    x_val_seq: np.ndarray
    x_eval_seq: np.ndarray
    train_threshold: float
    x_standardizer: Standardizer
    eval_split_name: str


@dataclass
class ClassificationMetrics:
    model_name: str
    eval_split: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    balanced_accuracy: float
    positive_rate_train: float
    positive_rate_eval: float
    score_threshold: float
    train_seconds: float
    inference_seconds: float
    train_subset: int
    val_subset: int
    eval_subset: int
    quantile: float
    seed: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_npy_pair(data_dir: Path, x_name: str, y_name: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path = data_dir / x_name
    y_path = data_dir / y_name
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Could not find required files: {x_path.name} and/or {y_path.name} in {data_dir}"
        )
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")


def maybe_load_npy_pair(data_dir: Path, x_name: str, y_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x_path = data_dir / x_name
    y_path = data_dir / y_name
    if not x_path.exists() or not y_path.exists():
        return None
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")


def select_indices(length: int, max_examples: int, seed: int) -> np.ndarray:
    if max_examples <= 0 or max_examples >= length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=max_examples, replace=False)).astype(np.int64)


def convert_flat_input_to_channels(x_row: np.ndarray) -> np.ndarray:
    """Convert one 124-d input vector to (6, 60), matching the current project layout."""
    x_row = np.asarray(x_row, dtype=np.float32)
    if x_row.shape[-1] != INPUT_DIM:
        raise ValueError(f"Expected input dim {INPUT_DIM}, got {x_row.shape[-1]}")
    temp = x_row[:LEVELS]
    qv = x_row[LEVELS : 2 * LEVELS]
    scalars = x_row[2 * LEVELS :]
    repeated_scalars = np.repeat(scalars[:, None], LEVELS, axis=1)
    return np.concatenate([temp[None, :], qv[None, :], repeated_scalars], axis=0).astype(np.float32)


def convert_flat_batch_to_channels(x_batch: np.ndarray) -> np.ndarray:
    x_batch = np.asarray(x_batch, dtype=np.float32)
    if x_batch.ndim != 2 or x_batch.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected shape (n, {INPUT_DIM}), got {x_batch.shape}")
    temp = x_batch[:, :LEVELS]
    qv = x_batch[:, LEVELS : 2 * LEVELS]
    scalars = x_batch[:, 2 * LEVELS :]
    repeated_scalars = np.repeat(scalars[:, :, None], LEVELS, axis=2)
    return np.concatenate([temp[:, None, :], qv[:, None, :], repeated_scalars], axis=1).astype(np.float32)


def convert_flat_batch_to_sequence(x_batch: np.ndarray) -> np.ndarray:
    channels = convert_flat_batch_to_channels(x_batch)
    return np.transpose(channels, (0, 2, 1)).astype(np.float32)


def total_precip_signal(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != TARGET_DIM:
        raise ValueError(f"Expected target shape (n, {TARGET_DIM}), got {y.shape}")
    return np.asarray(y[:, IDX_PRECSC] + y[:, IDX_PRECC], dtype=np.float32)


def make_binary_labels(
    y: np.ndarray,
    *,
    quantile: float = 0.9,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    signal = total_precip_signal(y)
    if threshold is None:
        threshold = float(np.quantile(signal, quantile))
    labels = (signal >= threshold).astype(np.int64)
    return labels, float(threshold)


def optimize_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (scores >= threshold).astype(np.int64)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def compute_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    y_pred = (scores >= threshold).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
    }


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def print_model_report(metrics: ClassificationMetrics, extra_lines: Optional[Sequence[str]] = None) -> None:
    print("=" * 72)
    print(metrics.model_name)
    if extra_lines:
        for line in extra_lines:
            print(line)
    print(f"eval split: {metrics.eval_split}")
    print(
        f"accuracy={metrics.accuracy:.4f} | f1={metrics.f1:.4f} | "
        f"precision={metrics.precision:.4f} | recall={metrics.recall:.4f} | "
        f"balanced_accuracy={metrics.balanced_accuracy:.4f}"
    )
    print(
        f"train time={format_seconds(metrics.train_seconds)} | "
        f"inference time={format_seconds(metrics.inference_seconds)} | "
        f"threshold={metrics.score_threshold:.2f}"
    )
    print(
        f"train positive rate={metrics.positive_rate_train:.4f} | "
        f"{metrics.eval_split} positive rate={metrics.positive_rate_eval:.4f}"
    )
    print(
        f"subsets: train={metrics.train_subset} | val={metrics.val_subset} | "
        f"eval={metrics.eval_subset} | quantile={metrics.quantile:.2f} | seed={metrics.seed}"
    )
    if metrics.notes:
        print(f"notes: {metrics.notes}")


def print_comparison_table(records: Sequence[ClassificationMetrics]) -> None:
    if not records:
        print("No results to compare.")
        return

    headers = [
        "Model",
        "Split",
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
        "Train",
        "Infer",
    ]
    rows = [
        [
            r.model_name,
            r.eval_split,
            f"{r.accuracy:.4f}",
            f"{r.f1:.4f}",
            f"{r.precision:.4f}",
            f"{r.recall:.4f}",
            f"{r.train_seconds:.2f}s",
            f"{r.inference_seconds:.2f}s",
        ]
        for r in records
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print("=" * 72)
    print("Combined comparison")
    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


def prepare_classification_data(
    *,
    data_dir: Path,
    train_subset: int,
    val_subset: int,
    eval_subset: int,
    quantile: float,
    seed: int,
) -> PreparedData:
    x_train_raw, y_train_raw = load_npy_pair(data_dir, "train_input.npy", "train_target.npy")
    x_val_raw, y_val_raw = load_npy_pair(data_dir, "val_input.npy", "val_target.npy")
    scoring = maybe_load_npy_pair(data_dir, "scoring_input.npy", "scoring_target.npy")

    train_idx = select_indices(x_train_raw.shape[0], train_subset, seed)
    val_idx = select_indices(x_val_raw.shape[0], val_subset, seed + 1)

    x_train = np.asarray(x_train_raw[train_idx], dtype=np.float32)
    y_train = np.asarray(y_train_raw[train_idx], dtype=np.float32)
    x_val = np.asarray(x_val_raw[val_idx], dtype=np.float32)
    y_val = np.asarray(y_val_raw[val_idx], dtype=np.float32)

    if scoring is None:
        x_eval = x_val.copy()
        y_eval = y_val.copy()
        eval_split_name = "val"
    else:
        x_scoring_raw, y_scoring_raw = scoring
        scoring_idx = select_indices(x_scoring_raw.shape[0], eval_subset, seed + 2)
        x_eval = np.asarray(x_scoring_raw[scoring_idx], dtype=np.float32)
        y_eval = np.asarray(y_scoring_raw[scoring_idx], dtype=np.float32)
        eval_split_name = "scoring"

    x_standardizer = Standardizer.fit(x_train)

    x_train_flat = x_standardizer.transform(x_train).astype(np.float32)
    x_val_flat = x_standardizer.transform(x_val).astype(np.float32)
    x_eval_flat = x_standardizer.transform(x_eval).astype(np.float32)

    y_train_cls, train_threshold = make_binary_labels(y_train, quantile=quantile)
    y_val_cls, _ = make_binary_labels(y_val, threshold=train_threshold)
    y_eval_cls, _ = make_binary_labels(y_eval, threshold=train_threshold)

    return PreparedData(
        x_train_flat=x_train_flat,
        y_train_cls=y_train_cls,
        x_val_flat=x_val_flat,
        y_val_cls=y_val_cls,
        x_eval_flat=x_eval_flat,
        y_eval_cls=y_eval_cls,
        x_train_seq=convert_flat_batch_to_sequence(x_train_flat),
        x_val_seq=convert_flat_batch_to_sequence(x_val_flat),
        x_eval_seq=convert_flat_batch_to_sequence(x_eval_flat),
        train_threshold=train_threshold,
        x_standardizer=x_standardizer,
        eval_split_name=eval_split_name,
    )
