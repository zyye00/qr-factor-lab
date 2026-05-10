import pandas as pd

from quant.data import (
    OHLCV_COLUMNS,
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
