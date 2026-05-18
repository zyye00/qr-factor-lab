from pathlib import Path

import pytest

from quant import cli


def test_data_cli_calls_data_pipeline(monkeypatch, capsys) -> None:
    calls = {}

    def fake_run_data_pipeline(config_path: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        return {"label_panel": Path("data/work/label_panel.parquet")}

    monkeypatch.setattr(cli, "run_data_pipeline", fake_run_data_pipeline)

    cli.main(["data", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "label_panel:" in output
    assert "label_panel.parquet" in output


def test_simulate_cli_calls_simulation_pipeline(monkeypatch, capsys) -> None:
    calls = {}

    def fake_run_simulation_pipeline(config_path: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        return {"final_report_template": Path("reports/final_report.md")}

    monkeypatch.setattr(cli, "run_simulation_pipeline", fake_run_simulation_pipeline)

    cli.main(["simulate", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "final_report_template:" in output
    assert "final_report.md" in output


def test_step_cli_calls_named_step(monkeypatch, capsys) -> None:
    calls = {}

    def fake_run_step(stage: str, config_path: str) -> dict[str, Path]:
        calls["stage"] = stage
        calls["config_path"] = config_path
        return {"factor_panel": Path("data/work/factor_panel.parquet")}

    monkeypatch.setattr(cli, "run_step", fake_run_step)

    cli.main(["step", "factors", "--config", "custom.yaml"])

    assert calls == {"stage": "factors", "config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "factor_panel:" in output
    assert "factor_panel.parquet" in output


def test_removed_legacy_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.main(["download-data"])
