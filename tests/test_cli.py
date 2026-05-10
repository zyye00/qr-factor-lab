from pathlib import Path

from quant import cli


def test_download_data_cli_calls_downloader(monkeypatch, capsys) -> None:
    calls = {}

    def fake_download_data(config_path: str, adjust: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        calls["adjust"] = adjust
        return {"stock_panel": Path("data/processed/stock_panel.parquet")}

    monkeypatch.setattr(cli, "download_data", fake_download_data)

    exit_code = cli.main(["download-data", "--config", "custom.yaml", "--adjust", ""])

    assert exit_code == 0
    assert calls == {"config_path": "custom.yaml", "adjust": ""}
    output = capsys.readouterr().out
    assert "stock_panel:" in output
    assert "stock_panel.parquet" in output


def test_preprocess_data_cli_calls_preprocessor(monkeypatch, capsys) -> None:
    calls = {}

    def fake_preprocess_data(config_path: str) -> Path:
        calls["config_path"] = config_path
        return Path("data/processed/clean_panel.parquet")

    monkeypatch.setattr(cli, "preprocess_data", fake_preprocess_data)

    exit_code = cli.main(["preprocess-data", "--config", "custom.yaml"])

    assert exit_code == 0
    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "clean_panel:" in output
    assert "clean_panel.parquet" in output
