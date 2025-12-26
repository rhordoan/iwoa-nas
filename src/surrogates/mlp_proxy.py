from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset


class MLPProxy(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ProxyMetrics:
    val_mae: float
    val_mse: float
    val_pearson: float


def _make_loaders(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader]:
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    train_ds: Dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds: Dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_mlp_proxy(
    csv_path: str | Path,
    *,
    output_path: str | Path = "results/surrogates/mlp_proxy.pth",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    hidden_dim: int = 64,
    seed: int = 42,
) -> ProxyMetrics:
    df = pd.read_csv(csv_path)
    feats = df[["depth_mult", "width_mult", "res_mult"]].values.astype(np.float32)
    labels = df["val_acc"].values.astype(np.float32)
    train_loader, val_loader = _make_loaders(feats, labels, batch_size, seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPProxy(hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad(set_to_none=True)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            opt.step()

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch)
    preds_np = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()
    val_mse = float(np.mean((preds_np - targets_np) ** 2))
    val_mae = float(np.mean(np.abs(preds_np - targets_np)))
    val_pearson = float(np.corrcoef(preds_np.flatten(), targets_np.flatten())[0, 1])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, output)

    return ProxyMetrics(val_mae=val_mae, val_mse=val_mse, val_pearson=val_pearson)


def load_mlp_proxy(path: str | Path, device: str | None = None) -> MLPProxy:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MLPProxy()
    checkpoint = torch.load(path, map_location=device_t)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device_t)
    model.eval()
    return model

