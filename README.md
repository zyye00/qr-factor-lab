# CSI500 Factor Lab

A small, reproducible research project for CSI 500 cross-sectional factor
analysis. The first milestone only establishes the project skeleton, shared
configuration, linting, and a smoke test.

## Scope

This project will gradually build:

- CSI 500 universe and OHLCV data ingestion
- basic return and forward-return labels
- simple cross-sectional factors
- IC / Rank IC evaluation
- quantile portfolio backtests
- transaction-cost and robustness checks
- a final Markdown research report

## Project Layout

```text
config.yaml
pyproject.toml
src/csi500_factor_lab/
tests/
```

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Checks

```bash
ruff check .
pytest
```

## Notes

This project is intended for research and educational purposes only. It does
not constitute investment advice.

The first-stage implementation uses current CSI 500 constituents instead of
historical index membership. Therefore, the backtest may suffer from
survivorship bias.

The transaction cost model is simplified and does not fully capture market
impact, limit-up/limit-down constraints, or intraday liquidity.

Long-short portfolio results are used to evaluate factor ranking power. They do
not directly represent an executable A-share trading strategy due to
short-selling constraints.
