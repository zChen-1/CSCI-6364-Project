from __future__ import annotations
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple
from torch import nn
from torch.utils.data import DataLoader
from data_utils import final_project_dataset, Standardizer, load_npy_pair
from models import ColumnCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-subset", type=int, default=4096)
    parser.add_argument("--val-subset", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_indices(length: int, max_examples: int, seed: int) -> np.ndarray:
    if max_examples <= 0 or max_examples >= length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=max_examples, replace=False)).astype(np.int64)


def fit_standardizers(x_train: np.ndarray, y_train: np.ndarray, train_idx: np.ndarray) -> Tuple[Standardizer, Standardizer]:
    x_sample = np.asarray(x_train[train_idx], dtype=np.float32)
    y_sample = np.asarray(y_train[train_idx], dtype=np.float32)
    return Standardizer.fit(x_sample), Standardizer.fit(y_sample)


def make_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, Standardizer, Standardizer]:
    x_train, y_train = load_npy_pair(args.data_dir, "train_input.npy", "train_target.npy")
    x_val, y_val = load_npy_pair(args.data_dir, "val_input.npy", "val_target.npy")
    train_idx = select_indices(x_train.shape[0], args.train_subset, args.seed)
    val_idx = select_indices(x_val.shape[0], args.val_subset, args.seed + 1)

    x_standardizer, y_standardizer = fit_standardizers(x_train, y_train, train_idx)
    train_ds = final_project_dataset(x_train, y_train, x_standardizer, y_standardizer, indices=train_idx)
    val_ds = final_project_dataset(x_val, y_val, x_standardizer, y_standardizer, indices=val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, x_standardizer, y_standardizer


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    groups = {
        "heating": slice(0, 60),
        "moistening": slice(60, 120),
        "fluxes": slice(120, 128),
    }
    for name, slc in groups.items():
        group_true = y_true[:, slc]
        group_pred = y_pred[:, slc]
        metrics[f"{name}_mae"] = float(np.mean(np.abs(group_true - group_pred)))
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean((group_true - group_pred) ** 2)))
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, y_standardizer: Standardizer) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        preds.append(out.cpu().numpy())
        targets.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    y_pred = y_standardizer.inverse_transform(y_pred)
    y_true = y_standardizer.inverse_transform(y_true)
    return compute_metrics(y_true, y_pred)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device) -> float:
    model.train()
    running = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * xb.shape[0]
        count += xb.shape[0]
    return running / max(count, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, _, y_standardizer = make_dataloaders(args)
    model = ColumnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss()

    best_metrics: Dict[str, float] = {}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device, y_standardizer)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | val_rmse={val_metrics['rmse']:.4f} | val_r2={val_metrics['r2']:.4f}"
        )

    elapsed = time.time() - start
    best_metrics["elapsed_seconds"] = elapsed
    best_metrics["device"] = str(device)
    best_metrics["train_subset"] = args.train_subset
    best_metrics["val_subset"] = args.val_subset

    for key, value in best_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
