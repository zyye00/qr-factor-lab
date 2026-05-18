from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from quant.backtest import compute_quantile_backtest
from quant.bootstrap import compute_bootstrap_ic
from quant.costs import compute_cost_analysis
from quant.factors import compute_factors
from quant.fetch import download_data
from quant.labels import compute_labels
from quant.metrics import compute_ic_analysis
from quant.preprocess import preprocess_data
from quant.report import generate_report

type StepResult = Path | Mapping[str, Path]
type StepRunner = Callable[[str], StepResult]

DATA_STEPS = ("download", "preprocess", "labels")
SIMULATION_STEPS = ("factors", "ic", "backtest", "costs", "bootstrap", "report")


@dataclass(frozen=True)
class Step:
    name: str
    output_name: str | None
    run: StepRunner

    def execute(self, config_path: str) -> dict[str, Path]:
        result = self.run(config_path)
        if isinstance(result, Path):
            return {self.output_name or self.name: result}
        return dict(result)


STEPS: dict[str, Step] = {
    "download": Step("download", None, download_data),
    "preprocess": Step("preprocess", "clean_panel", preprocess_data),
    "labels": Step("labels", "label_panel", compute_labels),
    "factors": Step("factors", "factor_panel", compute_factors),
    "ic": Step("ic", None, compute_ic_analysis),
    "backtest": Step("backtest", None, compute_quantile_backtest),
    "costs": Step("costs", None, compute_cost_analysis),
    "bootstrap": Step("bootstrap", None, compute_bootstrap_ic),
    "report": Step("report", None, generate_report),
}


def available_steps() -> tuple[str, ...]:
    return tuple(STEPS)


def run_data_pipeline(config_path: str = "config.yaml") -> dict[str, Path]:
    return run_steps(DATA_STEPS, config_path=config_path)


def run_simulation_pipeline(config_path: str = "config.yaml") -> dict[str, Path]:
    return run_steps(SIMULATION_STEPS, config_path=config_path)


def run_step(name: str, config_path: str = "config.yaml") -> dict[str, Path]:
    try:
        step = STEPS[name]
    except KeyError as error:
        available = ", ".join(available_steps())
        raise ValueError(
            f"Unknown pipeline step '{name}'. Available: {available}."
        ) from error
    return step.execute(config_path)


def run_steps(
    names: Sequence[str],
    config_path: str = "config.yaml",
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in names:
        paths.update(run_step(name, config_path=config_path))
    return paths
