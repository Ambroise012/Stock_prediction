"""PyTorch forecasting model for the stock pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class StockLSTM(nn.Module):
    """Sequence model for multi-step log-return forecasting."""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden_size, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_out, _ = self.lstm(x)
        last_step = sequence_out[:, -1, :]
        return self.head(self.dropout(last_step))


def train_pytorch_lstm(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    Y_val: np.ndarray | None = None,
    *,
    horizon: int,
    hidden_size: int = 64,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> tuple[nn.Module, list[float]]:
    """Train a small LSTM model using the same supervised arrays as the GRU flow."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StockLSTM(
        input_size=X_train.shape[-1],
        horizon=horizon,
        hidden_size=hidden_size,
    ).to(device)

    train_tensor = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(Y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        avg_loss = running_loss / max(len(train_tensor), 1)
        history.append(avg_loss)

    if X_val is not None and Y_val is not None:
        model.eval()
        with torch.no_grad():
            val_x = torch.from_numpy(X_val.astype(np.float32)).to(device)
            val_y = torch.from_numpy(Y_val.astype(np.float32)).to(device)
            _ = model(val_x)
            _ = criterion(model(val_x), val_y)

    return model.cpu(), history


def save_lstm_model(model: nn.Module, path: str | Path) -> None:
    """Save model weights using a simple PyTorch checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_lstm_model(path: str | Path, input_size: int, horizon: int) -> nn.Module:
    """Load a saved PyTorch LSTM model."""
    model = StockLSTM(input_size=input_size, horizon=horizon)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_lstm(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return multi-horizon predictions for a batch or single sample."""
    tensor = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        preds = model(tensor)
    return preds.numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate a PyTorch LSTM model.")
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL")
    parser.add_argument("--model-path", default="models/lstm_model.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"PyTorch LSTM model ready for {args.ticker}. Use src.train or src.predict_stock to fit and infer.")
