"""Command-line entry point for training and inference."""
import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stock model or run an inference."
    )
    parser.add_argument(
        "action",
        choices=("train", "infer"),
        help="Operation to run: train a model or infer future prices.",
    )
    parser.add_argument("ticker", help="Yahoo Finance ticker, for example AAPL")
    return parser.parse_args()


def run_training(ticker: str) -> None:
    """Run the existing training module in the current uv environment."""
    subprocess.run([sys.executable, "-m", "src.train", ticker], check=True)
    print(f"Training completed for {ticker}.")


def run_inference(ticker: str) -> None:
    """Run inference and print the forecast as a compact terminal table."""
    from src.predict_stock import predict_stock

    _, predictions, last_price, metrics_path = predict_stock(ticker)
    print(f"Ticker: {ticker}")
    print(f"Last price: {last_price:.2f}")
    print("Horizon | Predicted price | Change (%)")
    print("--------|-----------------|-----------")
    for horizon, prediction in enumerate(predictions, start=1):
        change = (prediction - last_price) / last_price * 100
        print(f"+{horizon:>5} | {prediction:>15.2f} | {change:>9.2f}")
    print(f"Metrics: {metrics_path}")


def main() -> None:
    args = parse_args()
    ticker = args.ticker.strip().upper()
    if not ticker:
        raise SystemExit("Ticker cannot be empty.")

    if args.action == "train":
        run_training(ticker)
    else:
        run_inference(ticker)


if __name__ == "__main__":
    main()