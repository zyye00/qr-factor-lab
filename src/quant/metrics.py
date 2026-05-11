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

DEFAULT_ROLLING_WINDOW = 20
DEFAULT_REPORTS_DIR = "reports"


def compute_ic(
    factor_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    return _compute_correlation_panel(
        factor_panel,
        label_panel,
        factor_columns=factor_columns,
        label_columns=label_columns,
        method="pearson",
    )


def compute_rank_ic(
    factor_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    return _compute_correlation_panel(
        factor_panel,
        label_panel,
        factor_columns=factor_columns,
        label_columns=label_columns,
        method="spearman",
    )


def compute_icir(ic_values: pd.Series) -> float:
    clean = ic_values.dropna()
    std = clean.std()
    if clean.empty or pd.isna(std) or std == 0:
        return np.nan
    return clean.mean() / std


def compute_rolling_ic(
    ic_panel: pd.DataFrame,
    window: int = DEFAULT_ROLLING_WINDOW,
    min_periods: int | None = None,
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive.")
    if min_periods is None:
        min_periods = window
    return ic_panel.sort_index().rolling(window=window, min_periods=min_periods).mean()


def summarize_ic(
    ic_panel: pd.DataFrame,
    rank_ic_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    summary = _summarize_correlation_panel(ic_panel, prefix="ic")
    if rank_ic_panel is not None:
        rank_summary = _summarize_correlation_panel(rank_ic_panel, prefix="rank_ic")
        summary = summary.join(rank_summary, how="outer")
    summary.index.name = "factor_label"
    return summary.sort_index()


def compute_ic_analysis(config_path: str = "config.yaml") -> dict[str, Path]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_dir = Path(config["data"]["processed_dir"])
    reports_dir = _configured_reports_dir(config, config_file)
    figures_dir = reports_dir / "figures"
    factor_panel = pd.read_parquet(processed_dir / "factor_panel.parquet")
    label_panel = pd.read_parquet(processed_dir / "label_panel.parquet")

    factor_columns = _configured_factor_columns(config, factor_panel)
    label_columns = _configured_label_columns(config, label_panel)
    rolling_window = config.get("metrics", {}).get(
        "rolling_window",
        DEFAULT_ROLLING_WINDOW,
    )

    ic_panel = compute_ic(factor_panel, label_panel, factor_columns, label_columns)
    rank_ic_panel = compute_rank_ic(
        factor_panel,
        label_panel,
        factor_columns,
        label_columns,
    )
    rolling_ic = compute_rolling_ic(ic_panel, window=int(rolling_window))
    summary = summarize_ic(ic_panel, rank_ic_panel)

    ic_path = save_parquet(ic_panel, processed_dir / "ic_panel.parquet")
    rank_ic_path = save_parquet(rank_ic_panel, processed_dir / "rank_ic_panel.parquet")
    rolling_ic_path = save_parquet(rolling_ic, processed_dir / "rolling_ic.parquet")
    summary_path = processed_dir / "ic_summary.csv"
    summary.to_csv(summary_path)
    markdown_path = write_ic_summary_markdown(
        summary,
        reports_dir / "ic_summary.md",
        rolling_window=int(rolling_window),
    )
    rolling_ic_figure_path = plot_rolling_ic(
        rolling_ic,
        summary,
        figures_dir / "rolling_ic.png",
    )

    return {
        "ic_panel": ic_path,
        "rank_ic_panel": rank_ic_path,
        "rolling_ic": rolling_ic_path,
        "ic_summary": summary_path,
        "ic_summary_markdown": markdown_path,
        "rolling_ic_figure": rolling_ic_figure_path,
    }


def write_ic_summary_markdown(
    summary: pd.DataFrame,
    output_path: str | Path,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = _ic_summary_markdown_table(summary)
    content = (
        "# IC Summary\n\n"
        "IC is the daily cross-sectional Pearson correlation between factor "
        "values and forward-return labels. Rank IC uses Spearman correlation.\n\n"
        f"Rolling IC window: {rolling_window} trading days.\n\n"
        f"{table}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def plot_rolling_ic(
    rolling_ic: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str | Path,
    max_columns: int = 10,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected = _select_rolling_ic_columns(rolling_ic, summary, max_columns=max_columns)
    figure, axis = plt.subplots(figsize=(14, 8))
    rolling_ic[selected].plot(ax=axis, linewidth=1.1)
    axis.axhline(0, color="#111827", linewidth=1)
    axis.set_title("Rolling IC")
    axis.set_xlabel("date")
    axis.set_ylabel("rolling IC")
    axis.legend(
        [_short_pair_label(column) for column in selected],
        fontsize=8,
        loc="best",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _compute_correlation_panel(
    factor_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    factor_columns: Sequence[str] | None,
    label_columns: Sequence[str] | None,
    method: str,
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
        for label in labels:
            result[f"{factor}__{label}"] = _daily_correlation(
                aligned,
                factor,
                label,
                method,
            ).reindex(dates)

    return result.sort_index()


def _daily_correlation(
    aligned: pd.DataFrame,
    factor: str,
    label: str,
    method: str,
) -> pd.Series:
    values = {}
    for date, group in aligned[[factor, label]].groupby(level="date", sort=True):
        values[date] = _correlate_pair(group, factor, label, method)
    series = pd.Series(values, dtype="float64")
    series.index.name = "date"
    return series


def _correlate_pair(
    group: pd.DataFrame,
    factor: str,
    label: str,
    method: str,
) -> float:
    pair = group[[factor, label]].dropna()
    if len(pair) < 2:
        return np.nan
    if pair[factor].nunique() < 2 or pair[label].nunique() < 2:
        return np.nan
    return pair[factor].corr(pair[label], method=method)


def _summarize_correlation_panel(panel: pd.DataFrame, prefix: str) -> pd.DataFrame:
    counts = panel.count()
    positive_rates = panel.gt(0).sum() / counts.replace(0, np.nan)
    summary = pd.DataFrame(
        {
            f"{prefix}_mean": panel.mean(),
            f"{prefix}_std": panel.std(),
            f"{prefix}_ir": panel.apply(compute_icir),
            f"{prefix}_positive_rate": positive_rates,
            f"{prefix}_n_days": counts,
        }
    )
    return summary


def _ic_summary_markdown_table(summary: pd.DataFrame) -> str:
    columns = [
        "factor_label",
        "ic_mean",
        "ic_ir",
        "ic_positive_rate",
        "rank_ic_mean",
        "rank_ic_ir",
        "rank_ic_positive_rate",
        "ic_n_days",
    ]
    table = summary.reset_index().sort_values("ic_mean", ascending=False)
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                str(row["factor_label"]),
                _format_float(row["ic_mean"]),
                _format_float(row["ic_ir"]),
                _format_float(row["ic_positive_rate"]),
                _format_float(row["rank_ic_mean"]),
                _format_float(row["rank_ic_ir"]),
                _format_float(row["rank_ic_positive_rate"]),
                _format_count(row["ic_n_days"]),
            ]
        )
    return _markdown_table(columns, rows)


def _markdown_table(headers: Sequence[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_float(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _format_count(value: object) -> str:
    if pd.isna(value):
        return "0"
    return str(int(value))


def _select_rolling_ic_columns(
    rolling_ic: pd.DataFrame,
    summary: pd.DataFrame,
    max_columns: int,
) -> list[str]:
    if rolling_ic.empty:
        return []
    ranked = summary["ic_mean"].abs().sort_values(ascending=False)
    selected = [column for column in ranked.index if column in rolling_ic.columns]
    selected.extend(column for column in rolling_ic.columns if column not in selected)
    return selected[:max_columns]


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
