import logging
from collections.abc import Callable
from pathlib import Path

import akshare as ak
import pandas as pd
import yaml

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "turnover"]
PANEL_INDEX = ["date", "ticker"]
STOCK_ADJUST = "hfq"
CDR_TICKERS = {"689009"}
DOWNLOAD_LOG_NAME = "download.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGER = logging.getLogger(__name__)
OHLCVFetcher = Callable[[str, str, str], pd.DataFrame]

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
    failures: list[tuple[str, str]] = []
    for source_name, fetcher in _stock_ohlcv_sources(ticker):
        try:
            frame = fetcher(ticker, start_date, end_date)
        except Exception as exc:
            failures.append((source_name, str(exc)))
            LOGGER.warning(
                "Stock OHLCV source %s failed for %s; trying fallback: %s",
                source_name,
                ticker,
                exc,
            )
            continue

        if not frame.empty:
            if failures:
                failed_sources = ", ".join(source for source, _ in failures)
                LOGGER.info(
                    "Downloaded stock OHLCV for %s from %s after fallback from %s",
                    ticker,
                    source_name,
                    failed_sources,
                )
            return frame

        failures.append((source_name, "empty frame"))
        LOGGER.warning(
            "Stock OHLCV source %s returned no rows for %s; trying fallback",
            source_name,
            ticker,
        )

    failure_details = "; ".join(
        f"{source_name}: {message}" for source_name, message in failures
    )
    raise RuntimeError(
        f"No OHLCV data downloaded for stock {ticker}. Tried "
        f"{', '.join(source_name for source_name, _ in _stock_ohlcv_sources(ticker))}. "
        f"{failure_details}",
    )


def _fetch_stock_ohlcv_eastmoney(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = ak.stock_zh_a_hist(
        symbol=ticker,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=STOCK_ADJUST,
    )
    return _normalize_downloaded_ohlcv(
        raw,
        ticker,
        start_date,
        end_date,
        volume_multiplier=100,
    )


def _fetch_stock_ohlcv_cdr(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = ak.stock_zh_a_cdr_daily(
        symbol=_stock_symbol(ticker),
        start_date=start_date,
        end_date=end_date,
    )
    return _normalize_downloaded_ohlcv(
        raw,
        ticker,
        start_date,
        end_date,
        volume_multiplier=100,
    )


def _fetch_stock_ohlcv_sina(
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
    return _normalize_downloaded_ohlcv(raw, ticker, start_date, end_date)


def _stock_ohlcv_sources(ticker: str) -> list[tuple[str, OHLCVFetcher]]:
    sources: list[tuple[str, OHLCVFetcher]] = [
        ("stock_zh_a_hist", _fetch_stock_ohlcv_eastmoney),
    ]
    if ticker in CDR_TICKERS:
        sources.append(("stock_zh_a_cdr_daily", _fetch_stock_ohlcv_cdr))
    sources.append(("stock_zh_a_daily", _fetch_stock_ohlcv_sina))
    return sources


def _normalize_downloaded_ohlcv(
    raw: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    volume_multiplier: int = 1,
) -> pd.DataFrame:
    frame = _filter_dates(normalize_ohlcv(raw, ticker=ticker), start_date, end_date)
    if volume_multiplier != 1:
        frame["volume"] = frame["volume"] * volume_multiplier
    return _drop_empty_ohlcv_rows(frame)


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
    frame = _drop_empty_ohlcv_rows(
        _filter_dates(normalize_ohlcv(raw, ticker=symbol), start_date, end_date),
    )
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
    end_date = _resolve_end_date(data_config["end_date"])
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
    stock_panel_path = processed_dir / "stock_panel.parquet"
    existing_stock_panel = load_existing_price_panel(stock_panel_path)
    stock_frames = _fetch_many_stocks(
        tickers,
        start_date,
        end_date,
        existing_panel=existing_stock_panel,
    )
    stock_panel = _merge_price_panels(
        existing_stock_panel,
        stock_frames,
        tickers,
        start_date,
        end_date,
    )
    stock_panel_path = save_parquet(stock_panel, stock_panel_path)
    LOGGER.info(
        "Saved stock OHLCV panel with %s rows to %s",
        len(stock_panel),
        stock_panel_path,
    )

    benchmark_path = processed_dir / f"benchmark_{benchmark}.parquet"
    existing_benchmark_panel = load_existing_price_panel(benchmark_path)
    benchmark_frames = _fetch_missing_ohlcv(
        [benchmark],
        start_date,
        end_date,
        existing_benchmark_panel,
        fetch_benchmark_ohlcv,
        "benchmark",
    )
    benchmark_panel = _merge_price_panels(
        existing_benchmark_panel,
        benchmark_frames,
        [benchmark],
        start_date,
        end_date,
    )
    benchmark_path = save_parquet(benchmark_panel, benchmark_path)
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


def load_existing_price_panel(path: str | Path) -> pd.DataFrame | None:
    panel_path = Path(path)
    if not panel_path.exists():
        return None
    panel = _coerce_price_panel(pd.read_parquet(panel_path))
    LOGGER.info("Loaded existing OHLCV panel with %s rows from %s", len(panel), path)
    return panel


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
    existing_panel: pd.DataFrame | None = None,
) -> list[pd.DataFrame]:
    return _fetch_missing_ohlcv(
        tickers,
        start_date,
        end_date,
        existing_panel,
        fetch_stock_ohlcv,
        "stock",
    )


def _fetch_missing_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    existing_panel: pd.DataFrame | None,
    fetcher: OHLCVFetcher,
    label: str,
) -> list[pd.DataFrame]:
    requests = _missing_download_ranges(tickers, start_date, end_date, existing_panel)
    if not requests:
        LOGGER.info(
            "%s OHLCV panel already covers %s ticker(s) from %s to %s",
            label.capitalize(),
            len(tickers),
            start_date,
            end_date,
        )
        return []

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for position, (ticker, request_start, request_end) in enumerate(requests, start=1):
        LOGGER.info(
            "[%s/%s] Fetching %s OHLCV for %s from %s to %s",
            position,
            len(requests),
            label,
            ticker,
            request_start,
            request_end,
        )
        try:
            frame = fetcher(ticker, request_start, request_end)
        except Exception as exc:
            failures.append((ticker, str(exc)))
            LOGGER.warning(
                "Failed to download %s OHLCV for %s: %s",
                label,
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
    if not frames and not _has_existing_price_data(
        existing_panel,
        tickers,
        start_date,
        end_date,
    ):
        raise RuntimeError(f"No {label} OHLCV data was downloaded.")
    return frames


def _merge_price_panels(
    existing_panel: pd.DataFrame | None,
    downloaded_frames: list[pd.DataFrame],
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    existing = _filter_price_panel(existing_panel, tickers, start_date, end_date)
    if not downloaded_frames:
        if existing.empty:
            raise RuntimeError("No OHLCV data is available for the requested range.")
        return existing

    downloaded = _coerce_price_panel(make_price_panel(downloaded_frames))
    if existing.empty:
        return _filter_price_panel(downloaded, tickers, start_date, end_date)

    combined_index = existing.index.union(downloaded.index)
    merged = downloaded.reindex(combined_index).combine_first(
        existing.reindex(combined_index),
    )
    return _filter_price_panel(merged, tickers, start_date, end_date)


def _missing_download_ranges(
    tickers: list[str],
    start_date: str,
    end_date: str,
    existing_panel: pd.DataFrame | None,
) -> list[tuple[str, str, str]]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if start > end:
        return []

    if existing_panel is None or existing_panel.empty:
        return [(ticker, start_date, end_date) for ticker in tickers]

    existing = _coerce_price_panel(existing_panel).reset_index()
    existing = existing[
        existing["ticker"].isin(tickers)
        & (existing["date"] >= start)
        & (existing["date"] <= end)
    ]

    requests: list[tuple[str, str, str]] = []
    latest_by_ticker = existing.groupby("ticker")["date"].max()
    for ticker in tickers:
        if ticker not in latest_by_ticker:
            requests.append((ticker, start_date, end_date))
            continue

        next_date = latest_by_ticker[ticker] + pd.Timedelta(days=1)
        if next_date <= end:
            requests.append((ticker, _format_ak_date(next_date), end_date))
    return requests


def _has_existing_price_data(
    existing_panel: pd.DataFrame | None,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> bool:
    return not _filter_price_panel(existing_panel, tickers, start_date, end_date).empty


def _filter_price_panel(
    panel: pd.DataFrame | None,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if panel is None or panel.empty:
        return _empty_price_panel()

    prices = _coerce_price_panel(panel).reset_index()
    prices = prices[prices["ticker"].isin(tickers)]
    prices = _filter_dates(prices, start_date, end_date)
    return _coerce_price_panel(prices)


def _coerce_price_panel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_price_panel()

    prices = frame.reset_index()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.zfill(6)
    for column in OHLCV_COLUMNS:
        if column not in prices.columns:
            prices[column] = pd.NA
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    prices = _drop_empty_ohlcv_rows(prices[PANEL_INDEX + OHLCV_COLUMNS])
    if prices.empty:
        return _empty_price_panel()
    return (
        prices.groupby(PANEL_INDEX, as_index=False)
        .last()
        .set_index(PANEL_INDEX)
        .sort_index()
    )


def _empty_price_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays([[], []], names=PANEL_INDEX)
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=index)


def _drop_empty_ohlcv_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=OHLCV_COLUMNS, how="all")


def _filter_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return frame[(frame["date"] >= start) & (frame["date"] <= end)]


def _resolve_end_date(end_date: str | None) -> str:
    if end_date:
        return end_date.replace("-", "")
    return pd.Timestamp.today().strftime("%Y%m%d")


def _format_ak_date(date: pd.Timestamp) -> str:
    return date.strftime("%Y%m%d")


def _stock_symbol(ticker: str) -> str:
    market = "sh" if ticker.startswith(("5", "6", "9")) else "sz"
    return f"{market}{ticker}"
