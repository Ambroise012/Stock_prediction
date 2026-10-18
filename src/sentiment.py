"""Text collection and daily financial sentiment features."""
import logging
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def fetch_google_news_rss(ticker: str, max_items: int = 50) -> pd.DataFrame:
    """Download recent headlines for a ticker from Google News RSS.

    This avoids the need for an API key while keeping the output in the format the
    project expects: `timestamp` and `text`.
    """
    q = quote(f'{ticker} stock OR {ticker} earnings OR "{ticker}"')
    url = (
        "https://news.google.com/rss/search?q="
        f"{q}&hl=en-US&gl=US&ceid=US:en"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    rows = []
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        pub_date = item.findtext("pubDate")
        if not title:
            continue
        if pub_date:
            parsed = parsedate_to_datetime(pub_date)
            timestamp = parsed.isoformat()
        else:
            timestamp = pd.Timestamp.utcnow().isoformat()
        rows.append({"timestamp": timestamp, "text": title})

    frame = pd.DataFrame(rows, columns=["timestamp", "text"])
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=["text", "timestamp"]).reset_index(drop=True)


def save_sentiment_csv(ticker: str, text_dir: str = "data/text", max_items: int = 50) -> Path:
    """Fetch recent headlines and save them in the expected per-ticker CSV format."""
    output_dir = Path(text_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker}.csv"
    frame = fetch_google_news_rss(ticker, max_items=max_items)
    if frame.empty:
        frame = pd.DataFrame(columns=["timestamp", "text"])
    frame.to_csv(path, index=False)
    logger.info("Saved %s rows for %s to %s", len(frame), ticker, path)
    return path


def _load_text_file(ticker: str, text_dir: str) -> pd.DataFrame:
    """Load optional historical text data from CSV.

    The CSV must contain a timestamp column and either text or title. Reddit
    exports can therefore be used without coupling the training code to PRAW.
    """
    path = Path(text_dir) / f"{ticker}.csv"
    if not path.exists():
        logger.info(
            "No historical text file found for %s at %s; sentiment features are neutral",
            ticker,
            path,
        )
        return pd.DataFrame(columns=["timestamp", "text"])

    frame = pd.read_csv(path)
    text_column = "text" if "text" in frame.columns else "title"
    if "timestamp" not in frame.columns or text_column not in frame.columns:
        raise ValueError(f"{path} must contain timestamp and text/title columns")
    return frame[["timestamp", text_column]].rename(columns={text_column: "text"})


def _score_with_finbert(texts: pd.Series) -> pd.Series:
    """Score text with FinBERT, keeping the dependency optional."""
    try:
        from transformers import pipeline

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            framework="pt",
        )
    except Exception as exc:
        logger.warning(
            "FinBERT sentiment unavailable (%s); sentiment features will be neutral",
            exc,
        )
        return pd.Series(0.0, index=texts.index)

    def signed_score(text: str) -> float:
        try:
            predictions = sentiment_pipeline(text, return_all_scores=True)[0]
            scores_by_label = {
                prediction["label"].lower(): prediction["score"]
                for prediction in predictions
            }
            return scores_by_label.get("positive", 0.0) - scores_by_label.get(
                "negative", 0.0
            )
        except Exception:
            return 0.0

    scores = texts.fillna("").astype(str).apply(signed_score)
    return scores


def process_and_debug_sentiment(ticker: str, text_dir: str, debug_file: str) -> None:
    """Process sentiment and save debug file."""
    # Load text data
    data = _load_text_file(ticker, text_dir)

    if data.empty:
        logger.info("No data to process for ticker: %s", ticker)
        return

    # Score sentiment
    data["sentiment"] = _score_with_finbert(data["text"])

    # Save debug file
    debug_path = Path(debug_file)
    data.to_csv(debug_path, index=False)
    logger.info("Debug file saved to %s", debug_path)
    

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.sentiment <TICKER> [max_items]")
        raise SystemExit(1)

    ticker = sys.argv[1].upper()
    max_items = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    path = save_sentiment_csv(ticker, max_items=max_items)
    print(f"Saved sentiment CSV: {path}")


def build_sentiment_features(
    price_index: pd.Index,
    ticker: str,
    text_dir: str = "data/text",
) -> pd.DataFrame:
    """Return leakage-safe daily sentiment aligned to a market-date index.

    Text published on day t is shifted to day t+1, so it cannot use information
    that was unavailable at the prediction time for day t.
    """
    empty = pd.DataFrame(
        {
            "sentiment_score": pd.Series(0.0, index=price_index, dtype=float),
            "sentiment_volume": pd.Series(0.0, index=price_index, dtype=float),
            "sentiment_text": pd.Series("", index=price_index, dtype=object),
        },
        index=price_index,
    )
    texts = _load_text_file(ticker, text_dir)
    if texts.empty:
        return empty

    texts["timestamp"] = pd.to_datetime(texts["timestamp"], utc=True, errors="coerce")
    texts["text"] = texts["text"].fillna("").astype(str)
    texts = texts.dropna(subset=["timestamp"])
    texts = texts[texts["text"].str.strip().ne("")].copy()
    if texts.empty:
        return empty

    texts["published_date"] = texts["timestamp"].dt.tz_convert(None).dt.normalize()
    texts["sentiment"] = _score_with_finbert(texts["text"])
    daily = texts.groupby("published_date").agg(
        sentiment_score=("sentiment", "mean"),
        sentiment_volume=("sentiment", "size"),
        sentiment_text=("text", lambda values: "\n".join(values)),
    )
    daily.index = pd.to_datetime(daily.index).normalize()
    result = empty.copy()
    result.index = pd.DatetimeIndex(result.index).tz_localize(None).normalize()
    trading_dates = result.index
    target_positions = trading_dates.searchsorted(daily.index, side="right")
    valid = target_positions < len(trading_dates)
    for source_position, target_position in enumerate(target_positions[valid]):
        result.iat[target_position, 0] = daily.iloc[source_position]["sentiment_score"]
        result.iat[target_position, 1] = daily.iloc[source_position]["sentiment_volume"]
        result.iat[target_position, 2] = daily.iloc[source_position]["sentiment_text"]
    return result[["sentiment_score", "sentiment_volume", "sentiment_text"]]