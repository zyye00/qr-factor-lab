import logging
from pathlib import Path

import akshare as ak
import pandas as pd
import yaml

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "turnover"]
PANEL_INDEX = ["date", "ticker"]
STOCK_ADJUST = "hfq"
DOWNLOAD_LOG_NAME = "download.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGER = logging.getLogger(__name__)

COLUMN_MAP = {
    "日期": "date",
    "股票代码": "ticker",
    "成分券代码": "ticker",
    "成分券名称": "name",
    "交易所": "exchange",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}


def normalize_universe(raw: pd.DataFrame) -> pd.DataFrame:
    universe = raw.rename(columns=COLUMN_MAP).copy()
    keep_columns = [
        column
        for column in ["date", "ticker", "name", "exchange"]
        if column in universe.columns
    ]
    universe = universe[keep_columns]
    universe["date"] = pd.to_datetime(universe["date"])
    universe["ticker"] = universe["ticker"].astype(str).str.zfill(6)
    return universe.sort_values("ticker").reset_index(drop=True)


def normalize_ohlcv(raw: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    prices = raw.rename(columns=COLUMN_MAP).copy()
    if ticker is not None:
        prices["ticker"] = ticker

    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.zfill(6)
    for column in OHLCV_COLUMNS:
        if column not in prices.columns:
            prices[column] = pd.NA
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    return prices[PANEL_INDEX + OHLCV_COLUMNS].sort_values(PANEL_INDEX)


def make_price_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one OHLCV frame is required.")
    panel = pd.concat(frames, ignore_index=True)
    return panel.set_index(PANEL_INDEX).sort_index()


def fetch_stock_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = ak.stock_zh_a_daily(
        symbol=_stock_symbol(ticker),
        start_date=start_date,
        end_date=end_date,
        adjust=STOCK_ADJUST,
    )
    frame = _filter_dates(normalize_ohlcv(raw, ticker=ticker), start_date, end_date)
    if frame.empty:
        raise RuntimeError(f"No OHLCV data downloaded for stock {ticker}.")
    return frame


def fetch_benchmark_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = ak.stock_zh_index_daily_tx(
        symbol=f"sh{symbol}",
        start_date=start_date,
        end_date=end_date,
    )
    frame = _filter_dates(normalize_ohlcv(raw, ticker=symbol), start_date, end_date)
    if frame.empty:
        raise RuntimeError(f"No OHLCV data downloaded for benchmark {symbol}.")
    return frame


def save_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path)
    return output_path


def download_data(
    config_path: str = "config.yaml",
) -> dict[str, Path]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_config = config["data"]
    start_date = data_config["start_date"].replace("-", "")
    end_date = (data_config["end_date"] or "20500101").replace("-", "")
    benchmark = data_config["benchmark"]
    raw_dir = Path(data_config["raw_dir"])
    processed_dir = Path(data_config["processed_dir"])
    log_path = configure_download_file_logging(raw_dir)

    LOGGER.info(
        "Starting data download: benchmark=%s start_date=%s end_date=%s "
        "stock_adjust=%s",
        benchmark,
        start_date,
        end_date,
        STOCK_ADJUST,
    )
    LOGGER.info("Writing download log to %s", log_path)
    universe = normalize_universe(ak.index_stock_cons_csindex(symbol=benchmark))
    universe_path = save_parquet(universe, raw_dir / "csi500_universe.parquet")
    LOGGER.info(
        "Saved CSI 500 universe with %s tickers to %s",
        len(universe),
        universe_path,
    )

    tickers = universe["ticker"].tolist()
    stock_frames = _fetch_many_stocks(tickers, start_date, end_date)
    stock_panel = make_price_panel(stock_frames)
    stock_panel_path = save_parquet(stock_panel, processed_dir / "stock_panel.parquet")
    LOGGER.info(
        "Saved stock OHLCV panel with %s rows to %s",
        len(stock_panel),
        stock_panel_path,
    )

    benchmark_frame = fetch_benchmark_ohlcv(benchmark, start_date, end_date)
    benchmark_panel = make_price_panel([benchmark_frame])
    benchmark_path = save_parquet(
        benchmark_panel,
        processed_dir / f"benchmark_{benchmark}.parquet",
    )
    LOGGER.info(
        "Saved benchmark OHLCV panel with %s rows to %s",
        len(benchmark_panel),
        benchmark_path,
    )

    return {
        "download_log": log_path,
        "universe": universe_path,
        "stock_panel": stock_panel_path,
        "benchmark": benchmark_path,
    }


def configure_download_file_logging(raw_dir: str | Path) -> Path:
    log_path = Path(raw_dir) / DOWNLOAD_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in list(LOGGER.handlers):
        if getattr(handler, "_qr_download_file_handler", False):
            LOGGER.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler._qr_download_file_handler = True
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    return log_path


def _fetch_many_stocks(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for position, ticker in enumerate(tickers, start=1):
        LOGGER.info(
            "[%s/%s] Fetching stock OHLCV for %s",
            position,
            len(tickers),
            ticker,
        )
        try:
            frame = fetch_stock_ohlcv(ticker, start_date, end_date)
        except Exception as exc:
            failures.append((ticker, str(exc)))
            LOGGER.warning(
                "Failed to download stock OHLCV for %s: %s",
                ticker,
                exc,
                exc_info=True,
            )
            continue
        if not frame.empty:
            frames.append(frame)

    if failures:
        failed = ", ".join(ticker for ticker, _ in failures[:10])
        LOGGER.warning("Skipped %s failed ticker(s): %s", len(failures), failed)
    if not frames:
        raise RuntimeError("No stock OHLCV data was downloaded.")
    return frames


def _filter_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return frame[(frame["date"] >= start) & (frame["date"] <= end)]


def _stock_symbol(ticker: str) -> str:
    market = "sh" if ticker.startswith(("5", "6", "9")) else "sz"
    return f"{market}{ticker}"
