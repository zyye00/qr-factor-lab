# CSI500 Factor Lab

A small, reproducible research project for CSI 500 cross-sectional factor
analysis. The current implementation can download the current CSI 500 universe,
stock OHLCV data, and benchmark OHLCV data to parquet files.

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
src/quant/
tests/
```

## CLI Workflow

Install the project in editable mode first:

```bash
python -m pip install -e ".[dev]"
```

The CLI is split into two main workflows:

```bash
quant data
quant simulate
```

`quant data` prepares the reusable market-data layer. `quant simulate` consumes
that layer and runs the factor research workflow.

## Data Pipeline

Build source data, the cleaned panel, and forward-return labels:

```bash
quant data
```

Outputs:

```text
data/download.log
data/source/stock_ohlcv.parquet
data/source/benchmark_000905_ohlcv.parquet
data/work/clean_panel.parquet
data/work/label_panel.parquet
```

The cleaned panel removes duplicate `date` + `ticker` rows, rows with missing
or non-positive `close`, and rows with `volume <= 0`. It adds `ret_1d`,
`ret_5d`, and `ret_20d`.

The label panel contains `fwd_excess_ret_5d` and `fwd_excess_ret_20d`, aligned to
the same `date` + `ticker` row as the factor observation. Tail rows without a
full forward horizon are kept as `NaN` so downstream evaluation can drop them
explicitly.

## Factor Simulation

After `quant data`, run the factor simulation workflow:

```bash
quant simulate
```

Outputs:

```text
data/work/factor_panel.parquet
data/work/ic_panel.parquet
data/work/rank_ic_panel.parquet
data/work/rolling_ic.parquet
data/work/ic_summary.parquet
data/work/quantile_returns.parquet
data/work/long_short_returns.parquet
data/work/long_only_returns.parquet
data/work/backtest_summary.parquet
reports/figures/quantile_cumulative_returns.png
data/work/long_short_turnover.parquet
data/work/long_only_turnover.parquet
data/work/cost_adjusted_long_short_returns.parquet
data/work/cost_adjusted_long_only_returns.parquet
data/work/cost_sensitivity_summary.parquet
reports/figures/cost_sensitivity.png
data/work/bootstrap_ic_summary.parquet
reports/tables/*.md
reports/figures/*.png
reports/final_report.md
```

The factor panel contains `reversal_5`, `momentum_20`, `low_volatility_20`,
`turnover_change_20`, and `liquidity_20`. Each factor is winsorized and z-scored
cross-sectionally by date.

IC is the daily cross-sectional Pearson correlation between a factor and a
forward-return label. Rank IC uses Spearman correlation on the same aligned
factor-label pairs.

Each factor is split into Q1-Q5 by date. The backtest reports average forward
label returns for each quantile, Q5-Q1 long-short returns, and Q5 long-only
returns. By default, `backtest.rebalance: label_horizon` samples each
factor-label portfolio on the label horizon: 5-day labels rebalance every 5
trading days, and 20-day labels rebalance every 20 trading days. The simulation
also evaluates transaction-cost sensitivity, bootstraps IC confidence intervals,
and regenerates report tables and figures.

The command does not overwrite an existing `reports/final_report.md`. The tables
and figures are regenerated from current analysis outputs and can be referenced
while writing the final report manually.

## Partial Steps

For debugging or partial rebuilds, run one explicit step:

```bash
quant step download
quant step preprocess
quant step labels
quant step factors
quant step ic
quant step backtest
quant step costs
quant step bootstrap
quant step report
```

## Checks

```bash
ruff check .
pytest
```

## CI

GitHub Actions runs the same checks on pushes and pull requests to `main`:

```bash
ruff check .
pytest
```

The CI does not download market data, so external data-source outages do not
block code checks.

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
