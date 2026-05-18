import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from quant.pipeline import (
    available_steps,
    run_data_pipeline,
    run_simulation_pipeline,
    run_step,
)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(prog="quant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser(
        "data",
        help="Download market data, preprocess it, and build labels.",
    )
    _add_config_argument(data_parser)

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run factor simulation, evaluation, robustness checks, and reports.",
    )
    _add_config_argument(simulate_parser)

    step_parser = subparsers.add_parser(
        "step",
        help="Run one pipeline step for debugging or partial rebuilds.",
    )
    step_parser.add_argument("stage", choices=available_steps())
    _add_config_argument(step_parser)

    args = parser.parse_args(argv)
    if args.command == "data":
        _print_paths(run_data_pipeline(config_path=args.config))
    elif args.command == "simulate":
        _print_paths(run_simulation_pipeline(config_path=args.config))
    elif args.command == "step":
        _print_paths(run_step(args.stage, config_path=args.config))


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.yaml")


def _print_paths(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        print(f"{name}: {path}")
