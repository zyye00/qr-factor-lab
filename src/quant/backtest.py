from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from quant.data import save_parquet
from quant.factors import FACTOR_COLUMNS
from quant.labels import LABEL_COLUMNS, align_factor_and_label

DEFAULT_REPORTS_DIR = "reports"


def assign_quantile_groups(
    factor_values: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    groups = factor_values.groupby(level="date", group_keys=False).apply(
        lambda values: _assign_daily_quantiles(values, n_quantiles)
    )
    groups.name = "quantile"
    return groups.sort_index()


def compute_quantile_returns(
    factor_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    factors = list(factor_columns or factor_panel.columns)
    labels = list(label_columns or label_panel.columns)
    aligned = align_factor_and_label(
        factor_panel[factors],
        label_panel[labels],
        dropna=False,
    )
    dates = pd.Index(aligned.index.get_level_values("date").unique()).sort_values()
    result = pd.DataFrame(index=dates)
    result.index.name = "date"

    for factor in factors:
        groups = assign_quantile_groups(aligned[factor], n_quantiles=n_quantiles)
        for label in labels:
            pair_returns = _daily_quantile_returns(
                labels=aligned[label],
                groups=groups,
                n_quantiles=n_quantiles,
            ).reindex(dates)
            for quantile in range(1, n_quantiles + 1):
                result[_quantile_column(factor, label, quantile)] = pair_returns[
                    quantile
                ]

    return result.sort_index()


def compute_long_short_returns(
    quantile_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    result = pd.DataFrame(index=quantile_returns.index)
    result.index.name = quantile_returns.index.name
    for pair in _quantile_pairs(quantile_returns):
        high = f"{pair}__Q{n_quantiles}"
        low = f"{pair}__Q1"
        if high in quantile_returns.columns and low in quantile_returns.columns:
            result[pair] = quantile_returns[high] - quantile_returns[low]
    return result.sort_index()


def compute_long_only_returns(
    quantile_returns: pd.DataFrame,
    long_quantile: int = 5,
) -> pd.DataFrame:
    result = pd.DataFrame(index=quantile_returns.index)
    result.index.name = quantile_returns.index.name
    suffix = f"__Q{long_quantile}"
    for column in quantile_returns.columns:
        if column.endswith(suffix):
            result[column.removesuffix(suffix)] = quantile_returns[column]
    return result.sort_index()


def compute_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns.fillna(0)).cumprod() - 1


def select_rebalance_dates(
    returns: pd.DataFrame,
    rebalance: str | None,
) -> pd.DataFrame:
    if rebalance is None or rebalance == "" or rebalance.upper() == "D":
        return returns.sort_index()
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex.")

    periods = returns.index.to_series().dt.to_period(rebalance)
    last_dates = returns.groupby(periods, group_keys=False).tail(1).index
    return returns.loc[last_dates].sort_index()


def summarize_backtest(
    long_short_returns: pd.DataFrame,
    long_only_returns: pd.DataFrame,
) -> pd.DataFrame:
    long_short_summary = _summarize_returns(long_short_returns, "long_short")
    long_only_summary = _summarize_returns(long_only_returns, "long_only")
    summary = long_short_summary.join(long_only_summary, how="outer")
    summary.index.name = "factor_label"
    return summary.sort_index()


def compute_quantile_backtest(config_path: str = "config.yaml") -> dict[str, Path]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_dir = Path(config["data"]["processed_dir"])
    reports_dir = _configured_reports_dir(config, config_file)
    figures_dir = reports_dir / "figures"
    n_quantiles = int(config.get("backtest", {}).get("n_quantiles", 5))
    rebalance = config.get("backtest", {}).get("rebalance")

    factor_panel = pd.read_parquet(processed_dir / "factor_panel.parquet")
    label_panel = pd.read_parquet(processed_dir / "label_panel.parquet")
    factor_columns = _configured_factor_columns(config, factor_panel)
    label_columns = _configured_label_columns(config, label_panel)

    quantile_returns = compute_quantile_returns(
        factor_panel,
        label_panel,
        factor_columns=factor_columns,
        label_columns=label_columns,
        n_quantiles=n_quantiles,
    )
    quantile_returns = select_rebalance_dates(quantile_returns, rebalance)
    long_short_returns = compute_long_short_returns(
        quantile_returns,
        n_quantiles=n_quantiles,
    )
    long_only_returns = compute_long_only_returns(
        quantile_returns,
        long_quantile=n_quantiles,
    )
    summary = summarize_backtest(long_short_returns, long_only_returns)

    quantile_path = save_parquet(
        quantile_returns,
        processed_dir / "quantile_returns.parquet",
    )
    long_short_path = save_parquet(
        long_short_returns,
        processed_dir / "long_short_returns.parquet",
    )
    long_only_path = save_parquet(
        long_only_returns,
        processed_dir / "long_only_returns.parquet",
    )
    summary_path = processed_dir / "backtest_summary.csv"
    summary.to_csv(summary_path)
    figure_path = plot_cumulative_returns(
        long_short_returns,
        long_only_returns,
        figures_dir / "quantile_cumulative_returns.png",
    )

    return {
        "quantile_returns": quantile_path,
        "long_short_returns": long_short_path,
        "long_only_returns": long_only_path,
        "backtest_summary": summary_path,
        "cumulative_return_figure": figure_path,
    }


def plot_cumulative_returns(
    long_short_returns: pd.DataFrame,
    long_only_returns: pd.DataFrame,
    output_path: str | Path,
    max_columns: int = 5,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    long_short_cumulative = compute_cumulative_returns(long_short_returns)
    long_only_cumulative = compute_cumulative_returns(long_only_returns)
    long_short_selected = _select_return_columns(long_short_cumulative, max_columns)
    long_only_selected = _select_return_columns(long_only_cumulative, max_columns)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
    if long_short_selected:
        long_short_cumulative[long_short_selected].plot(ax=axes[0], linewidth=1.1)
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_title("Q5-Q1 Long-Short Cumulative Returns")
    axes[0].set_ylabel("cumulative return")
    axes[0].legend(
        [_short_pair_label(column) for column in long_short_selected],
        fontsize=8,
        loc="best",
    )

    if long_only_selected:
        long_only_cumulative[long_only_selected].plot(ax=axes[1], linewidth=1.1)
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_title("Q5 Long-Only Cumulative Returns")
    axes[1].set_xlabel("date")
    axes[1].set_ylabel("cumulative return")
    axes[1].legend(
        [_short_pair_label(column) for column in long_only_selected],
        fontsize=8,
        loc="best",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _assign_daily_quantiles(values: pd.Series, n_quantiles: int) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    clean = values.dropna()
    if len(clean) < n_quantiles or clean.nunique() < n_quantiles:
        return result

    ranked = clean.rank(method="first")
    assigned = pd.qcut(
        ranked,
        q=n_quantiles,
        labels=range(1, n_quantiles + 1),
    )
    result.loc[clean.index] = assigned.astype("int64")
    return result


def _daily_quantile_returns(
    labels: pd.Series,
    groups: pd.Series,
    n_quantiles: int,
) -> pd.DataFrame:
    data = pd.DataFrame({"label": labels, "quantile": groups}).dropna()
    returns = (
        data.groupby([data.index.get_level_values("date"), "quantile"])["label"]
        .mean()
        .unstack("quantile")
    )
    return returns.reindex(columns=range(1, n_quantiles + 1))


def _summarize_returns(returns: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cumulative = compute_cumulative_returns(returns)
    counts = returns.count()
    summary = pd.DataFrame(
        {
            f"{prefix}_mean": returns.mean(),
            f"{prefix}_std": returns.std(),
            f"{prefix}_hit_rate": returns.gt(0).sum() / counts.replace(0, np.nan),
            f"{prefix}_cumulative_return": cumulative.iloc[-1],
            f"{prefix}_n_periods": counts,
        }
    )
    return summary


def _select_return_columns(returns: pd.DataFrame, max_columns: int) -> list[str]:
    if returns.empty:
        return []
    final_returns = returns.iloc[-1].abs().sort_values(ascending=False)
    return [column for column in final_returns.index[:max_columns]]


def _quantile_pairs(quantile_returns: pd.DataFrame) -> list[str]:
    pairs = {column.rsplit("__Q", maxsplit=1)[0] for column in quantile_returns.columns}
    return sorted(pairs)


def _quantile_column(factor: str, label: str, quantile: int) -> str:
    return f"{factor}__{label}__Q{quantile}"


def _short_pair_label(label: str) -> str:
    return label.replace("__fwd_excess_ret_", " -> ").replace("__fwd_ret_", " -> ")


def _configured_reports_dir(config: dict, config_path: Path) -> Path:
    reports_dir = Path(config.get("reports", {}).get("dir", DEFAULT_REPORTS_DIR))
    if reports_dir.is_absolute():
        return reports_dir
    return config_path.parent / reports_dir


def _configured_factor_columns(
    config: dict,
    factor_panel: pd.DataFrame,
) -> list[str]:
    configured = config.get("features", {}).get("factors", FACTOR_COLUMNS)
    return _require_columns(factor_panel, configured, "factor panel")


def _configured_label_columns(
    config: dict,
    label_panel: pd.DataFrame,
) -> list[str]:
    if "labels" not in config:
        return _require_columns(label_panel, LABEL_COLUMNS, "label panel")

    use_excess_return = config["labels"].get("use_excess_return", True)
    prefix = "fwd_excess_ret" if use_excess_return else "fwd_ret"
    configured = [
        f"{prefix}_{int(horizon)}d" for horizon in config["labels"]["horizons"]
    ]
    return _require_columns(label_panel, configured, "label panel")


def _require_columns(
    panel: pd.DataFrame,
    columns: Sequence[str],
    panel_name: str,
) -> list[str]:
    missing = [column for column in columns if column not in panel.columns]
    if missing:
        raise ValueError(f"Missing {panel_name} column(s): {', '.join(missing)}.")
    return list(columns)
