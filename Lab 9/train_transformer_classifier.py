'''
This model processes each sample by treating it as a sequence,
utilizing a self-attention mechanism to learn the interactions
between various positions within the sequence, and subsequently
predicts a class label for the entire sample. In this document,
the model is referred to as the "Climate Column Transformer," and
its associated training workflow encompasses data loading, network
training, selection of the optimal validation threshold, and
the reporting of final classification metrics.
'''

from __future__ import annotations
import os
import time
from copy import deepcopy
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab9_config import (
    DATA_DIR,
    EVAL_SUBSET,
    QUANTILE,
    SEED,
    TRAIN_SUBSET,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_LEARNING_RATE,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_WEIGHT_DECAY,
    VAL_SUBSET,
)
from lab9_utils import (
    ClassificationMetrics,
    compute_classification_metrics,
    optimize_threshold,
    prepare_classification_data,
    print_model_report,
)


class ClimateColumnTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 6,  # Each position in the sequence has 6 features
        seq_len: int = 60,  # The sequence length is 60
        d_model: int = 64,  # The internal transformer embedding size defaults to 64
        nhead: int = 4,  # Uses 4 attention heads
        num_layers: int = 2,  # Uses 2 transformer encoder layers
        dim_feedforward: int = 128,  # The feed-forward network inside each encoder block has hidden size 128
        dropout: float = 0.1,  # Uses dropout probability 0.1
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) + self.pos_embedding
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x).squeeze(-1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def evaluate(model: nn.Module, loader: DataLoader, y_true: np.ndarray, device: torch.device, threshold: float) -> Dict[str, float]:
    scores = predict_scores(model, loader, device)
    return compute_classification_metrics(y_true, scores, threshold=threshold)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_transformer(print_output: bool = True) -> ClassificationMetrics:
    set_seed(SEED)
    device = torch.device(DEVICE)

    if device.type == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    data = prepare_classification_data(
        data_dir=DATA_DIR,
        train_subset=TRAIN_SUBSET,
        val_subset=VAL_SUBSET,
        eval_subset=EVAL_SUBSET,
        quantile=QUANTILE,
        seed=SEED,
    )

    train_loader = make_loader(data.x_train_seq, data.y_train_cls, TRANSFORMER_BATCH_SIZE, shuffle=True)
    val_loader = make_loader(data.x_val_seq, data.y_val_cls, TRANSFORMER_BATCH_SIZE, shuffle=False)
    eval_loader = make_loader(data.x_eval_seq, data.y_eval_cls, TRANSFORMER_BATCH_SIZE, shuffle=False)

    model = ClimateColumnTransformer(
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
        dropout=TRANSFORMER_DROPOUT,
    ).to(device)

    pos = int(data.y_train_cls.sum())
    neg = int(len(data.y_train_cls) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRANSFORMER_LEARNING_RATE, weight_decay=TRANSFORMER_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_threshold = 0.5
    best_val_f1 = -1.0
    train_start = time.time()

    for epoch in range(1, TRANSFORMER_EPOCHS + 1):
        model.train()
        running = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]

        val_scores = predict_scores(model, val_loader, device)
        threshold, val_f1 = optimize_threshold(data.y_val_cls, val_scores)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = threshold
            best_state = deepcopy(model.state_dict())

        train_loss = running / max(count, 1)
        if print_output:
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_best_f1={val_f1:.4f} | threshold={threshold:.2f}")

    train_seconds = time.time() - train_start
    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    model.load_state_dict(best_state)

    infer_start = time.time()
    eval_scores = predict_scores(model, eval_loader, device)
    infer_seconds = time.time() - infer_start
    score_metrics = compute_classification_metrics(data.y_eval_cls, eval_scores, threshold=best_threshold)

    metrics = ClassificationMetrics(
        model_name="Transformer",
        eval_split=data.eval_split_name,
        accuracy=score_metrics["accuracy"],
        f1=score_metrics["f1"],
        precision=score_metrics["precision"],
        recall=score_metrics["recall"],
        balanced_accuracy=score_metrics["balanced_accuracy"],
        positive_rate_train=float(data.y_train_cls.mean()),
        positive_rate_eval=float(data.y_eval_cls.mean()),
        score_threshold=best_threshold,
        train_seconds=float(train_seconds),
        inference_seconds=float(infer_seconds),
        train_subset=int(data.x_train_seq.shape[0]),
        val_subset=int(data.x_val_seq.shape[0]),
        eval_subset=int(data.x_eval_seq.shape[0]),
        quantile=float(QUANTILE),
        seed=int(SEED),
        notes=(
            f"device={device.type}; d_model={TRANSFORMER_D_MODEL}; nhead={TRANSFORMER_NHEAD}; "
            f"num_layers={TRANSFORMER_NUM_LAYERS}; best_val_f1={best_val_f1:.4f}"
        ),
    )

    if print_output:
        print_model_report(
            metrics,
            extra_lines=[
                f"data_dir={DATA_DIR}",
                (
                    f"device={device.type} | epochs={TRANSFORMER_EPOCHS} | batch_size={TRANSFORMER_BATCH_SIZE} | "
                    f"lr={TRANSFORMER_LEARNING_RATE} | weight_decay={TRANSFORMER_WEIGHT_DECAY}"
                ),
                (
                    f"d_model={TRANSFORMER_D_MODEL} | nhead={TRANSFORMER_NHEAD} | "
                    f"num_layers={TRANSFORMER_NUM_LAYERS} | ff={TRANSFORMER_DIM_FEEDFORWARD} | "
                    f"dropout={TRANSFORMER_DROPOUT}"
                ),
                f"best val F1={best_val_f1:.4f}",
            ],
        )
    return metrics


def main() -> None:
    run_transformer(print_output=True)


if __name__ == "__main__":
    main()
