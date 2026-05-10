import argparse
from collections.abc import Sequence

from quant.data import download_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-data")
    download_parser.add_argument("--config", default="config.yaml")
    download_parser.add_argument("--adjust", default="qfq")

    args = parser.parse_args(argv)
    if args.command == "download-data":
        paths = download_data(
            config_path=args.config,
            adjust=args.adjust,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    return 0
