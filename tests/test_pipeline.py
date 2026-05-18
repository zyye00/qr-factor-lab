from pathlib import Path

from quant import pipeline


def test_data_pipeline_runs_data_steps_in_order(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        pipeline,
        "STEPS",
        {
            "download": _fake_step("download", calls),
            "preprocess": _fake_step("preprocess", calls),
            "labels": _fake_step("labels", calls),
        },
    )

    paths = pipeline.run_data_pipeline(config_path="custom.yaml")

    assert calls == [
        ("download", "custom.yaml"),
        ("preprocess", "custom.yaml"),
        ("labels", "custom.yaml"),
    ]
    assert list(paths) == ["download_path", "preprocess_path", "labels_path"]


def test_simulation_pipeline_runs_simulation_steps_in_order(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        pipeline,
        "STEPS",
        {
            "factors": _fake_step("factors", calls),
            "ic": _fake_step("ic", calls),
            "backtest": _fake_step("backtest", calls),
            "costs": _fake_step("costs", calls),
            "bootstrap": _fake_step("bootstrap", calls),
            "report": _fake_step("report", calls),
        },
    )

    paths = pipeline.run_simulation_pipeline(config_path="custom.yaml")

    assert calls == [
        ("factors", "custom.yaml"),
        ("ic", "custom.yaml"),
        ("backtest", "custom.yaml"),
        ("costs", "custom.yaml"),
        ("bootstrap", "custom.yaml"),
        ("report", "custom.yaml"),
    ]
    assert list(paths) == [
        "factors_path",
        "ic_path",
        "backtest_path",
        "costs_path",
        "bootstrap_path",
        "report_path",
    ]


def _fake_step(name: str, calls: list[tuple[str, str]]) -> pipeline.Step:
    def run(config_path: str) -> Path:
        calls.append((name, config_path))
        return Path(f"{name}.parquet")

    return pipeline.Step(name, f"{name}_path", run)
