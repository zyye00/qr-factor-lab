import pandas as pd

from quant import data
from quant.data import (
    DOWNLOAD_LOG_NAME,
    OHLCV_COLUMNS,
    STOCK_ADJUST,
    configure_download_file_logging,
    fetch_stock_ohlcv,
    make_price_panel,
    normalize_ohlcv,
    normalize_universe,
    save_parquet,
)


def test_normalize_ohlcv_uses_date_ticker_panel_fields() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "股票代码": ["1", "1"],
            "开盘": [10.0, 10.5],
            "最高": [11.0, 10.8],
            "最低": [9.8, 10.1],
            "收盘": [10.2, 10.6],
            "成交量": [1000, 1200],
            "成交额": [10200.0, 12720.0],
            "换手率": [0.5, 0.6],
        }
    )

    normalized = normalize_ohlcv(raw)
    panel = make_price_panel([normalized])

    assert panel.index.names == ["date", "ticker"]
    assert list(panel.columns) == OHLCV_COLUMNS
    assert panel.index.get_level_values("ticker").tolist() == ["000001", "000001"]


def test_normalize_universe_keeps_core_fields() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2026-05-08"],
            "成分券代码": ["9"],
            "成分券名称": ["中国宝安"],
            "交易所": ["深圳证券交易所"],
        }
    )

    universe = normalize_universe(raw)

    assert universe.loc[0, "ticker"] == "000009"
    assert list(universe.columns) == ["date", "ticker", "name", "exchange"]


def test_save_parquet_preserves_panel_index(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02"],
            "股票代码": ["000001"],
            "开盘": [10.0],
            "最高": [11.0],
            "最低": [9.8],
            "收盘": [10.2],
            "成交量": [1000],
            "成交额": [10200.0],
            "换手率": [0.5],
        }
    )
    panel = make_price_panel([normalize_ohlcv(raw)])

    path = save_parquet(panel, tmp_path / "panel.parquet")
    restored = pd.read_parquet(path)

    assert restored.index.names == ["date", "ticker"]
    assert list(restored.columns) == OHLCV_COLUMNS


def test_fetch_stock_ohlcv_always_uses_hfq_adjust(monkeypatch) -> None:
    calls = {}

    def fake_stock_zh_a_daily(
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        calls["symbol"] = symbol
        calls["adjust"] = adjust
        return pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000],
                "amount": [10200.0],
                "turnover": [0.5],
            }
        )

    monkeypatch.setattr(data.ak, "stock_zh_a_daily", fake_stock_zh_a_daily)

    fetch_stock_ohlcv("000001", "20240101", "20240131")

    assert calls == {"symbol": "sz000001", "adjust": STOCK_ADJUST}


def test_fetch_many_stocks_logs_download_failures(monkeypatch, caplog) -> None:
    def fake_fetch_stock_ohlcv(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if ticker == "000002":
            raise RuntimeError("network timeout")
        return pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "ticker": [ticker],
                "open": [10.0],
                "high": [11.0],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000],
                "amount": [10200.0],
                "turnover": [0.5],
            }
        )

    monkeypatch.setattr(data, "fetch_stock_ohlcv", fake_fetch_stock_ohlcv)

    with caplog.at_level("WARNING", logger="quant.data"):
        frames = data._fetch_many_stocks(["000001", "000002"], "20240101", "20240131")

    assert len(frames) == 1
    assert "Failed to download stock OHLCV for 000002" in caplog.text
    assert "Skipped 1 failed ticker(s): 000002" in caplog.text


def test_configure_download_file_logging_writes_to_raw_dir(tmp_path) -> None:
    log_path = configure_download_file_logging(tmp_path)

    data.LOGGER.info("sample download message")
    _close_download_file_handlers()

    assert log_path == tmp_path / DOWNLOAD_LOG_NAME
    assert "sample download message" in log_path.read_text(encoding="utf-8")


def _close_download_file_handlers() -> None:
    for handler in list(data.LOGGER.handlers):
        if getattr(handler, "_qr_download_file_handler", False):
            data.LOGGER.removeHandler(handler)
            handler.close()
