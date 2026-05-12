import pandas as pd
import pytest

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


def test_fetch_stock_ohlcv_prefers_eastmoney_hist_with_hfq_adjust(
    monkeypatch,
) -> None:
    calls = {}

    def fake_stock_zh_a_hist(
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        calls["symbol"] = symbol
        calls["period"] = period
        calls["adjust"] = adjust
        return pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "股票代码": ["000001"],
                "开盘": [10.0],
                "最高": [11.0],
                "最低": [9.8],
                "收盘": [10.2],
                "成交量": [10],
                "成交额": [10200.0],
                "换手率": [0.5],
            }
        )

    monkeypatch.setattr(data.ak, "stock_zh_a_hist", fake_stock_zh_a_hist)

    frame = fetch_stock_ohlcv("000001", "20240101", "20240131")

    assert calls == {"symbol": "000001", "period": "daily", "adjust": STOCK_ADJUST}
    assert frame.loc[0, "volume"] == 1000


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


def test_fetch_stock_ohlcv_falls_back_to_sina_daily(monkeypatch) -> None:
    calls = []

    def fake_stock_zh_a_hist(
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        calls.append(("hist", symbol, adjust))
        raise RuntimeError("eastmoney timeout")

    def fake_stock_zh_a_daily(
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        calls.append(("daily", symbol, adjust))
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

    monkeypatch.setattr(data.ak, "stock_zh_a_hist", fake_stock_zh_a_hist)
    monkeypatch.setattr(data.ak, "stock_zh_a_daily", fake_stock_zh_a_daily)

    frame = fetch_stock_ohlcv("000001", "20240101", "20240131")

    assert calls == [
        ("hist", "000001", STOCK_ADJUST),
        ("daily", "sz000001", STOCK_ADJUST),
    ]
    assert frame.loc[0, "close"] == 10.2


def test_fetch_stock_ohlcv_falls_back_to_cdr_source_for_689009(
    monkeypatch,
) -> None:
    calls = []

    def fake_stock_zh_a_hist(
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        calls.append(("hist", symbol))
        raise RuntimeError("eastmoney empty")

    def fake_stock_zh_a_cdr_daily(
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        calls.append(("cdr", symbol))
        return pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.8],
                "close": [10.2],
                "volume": [10],
            }
        )

    def fake_stock_zh_a_daily(
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        calls.append(("daily", symbol))
        return pd.DataFrame()

    monkeypatch.setattr(data.ak, "stock_zh_a_hist", fake_stock_zh_a_hist)
    monkeypatch.setattr(data.ak, "stock_zh_a_cdr_daily", fake_stock_zh_a_cdr_daily)
    monkeypatch.setattr(data.ak, "stock_zh_a_daily", fake_stock_zh_a_daily)

    frame = fetch_stock_ohlcv("689009", "20240101", "20240131")

    assert calls == [("hist", "689009"), ("cdr", "sh689009")]
    assert frame.loc[0, "ticker"] == "689009"
    assert frame.loc[0, "volume"] == 1000
    assert pd.isna(frame.loc[0, "amount"])


def test_fetch_many_stocks_downloads_only_missing_ticker_and_updates(
    monkeypatch,
) -> None:
    existing = make_price_panel(
        [
            _ohlcv_frame("2024-01-02", "000001", close=10.2),
            _ohlcv_frame("2024-01-03", "000001", close=10.6),
        ]
    )
    calls = []

    def fake_fetch_stock_ohlcv(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        calls.append((ticker, start_date, end_date))
        return _ohlcv_frame(start_date, ticker, close=11.0)

    monkeypatch.setattr(data, "fetch_stock_ohlcv", fake_fetch_stock_ohlcv)

    frames = data._fetch_many_stocks(
        ["000001", "000002"],
        "20240101",
        "20240105",
        existing_panel=existing,
    )

    assert len(frames) == 2
    assert calls == [
        ("000001", "20240104", "20240105"),
        ("000002", "20240101", "20240105"),
    ]


def test_merge_price_panels_does_not_overwrite_existing_values_with_na() -> None:
    existing = make_price_panel(
        [_ohlcv_frame("2024-01-02", "000001", open_=10.0, close=10.2)]
    )
    downloaded = _ohlcv_frame("2024-01-02", "000001", open_=11.0, close=pd.NA)

    merged = data._merge_price_panels(
        existing,
        [downloaded],
        ["000001"],
        "20240101",
        "20240131",
    )

    row = merged.loc[(pd.Timestamp("2024-01-02"), "000001")]
    assert row["open"] == 11.0
    assert row["close"] == 10.2


def test_fetch_stock_ohlcv_rejects_all_na_rows(monkeypatch) -> None:
    def fake_stock_zh_a_hist(
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        return pd.DataFrame({"date": ["2024-01-02"]})

    def fake_stock_zh_a_daily(
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        return pd.DataFrame({"date": ["2024-01-02"]})

    monkeypatch.setattr(data.ak, "stock_zh_a_hist", fake_stock_zh_a_hist)
    monkeypatch.setattr(data.ak, "stock_zh_a_daily", fake_stock_zh_a_daily)

    with pytest.raises(RuntimeError, match="No OHLCV data downloaded"):
        fetch_stock_ohlcv("000001", "20240101", "20240131")


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


def _ohlcv_frame(
    date: str,
    ticker: str,
    open_: object = 10.0,
    close: object = 10.2,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date)],
            "ticker": [ticker],
            "open": [open_],
            "high": [11.0],
            "low": [9.8],
            "close": [close],
            "volume": [1000],
            "amount": [10200.0],
            "turnover": [0.5],
        }
    )
