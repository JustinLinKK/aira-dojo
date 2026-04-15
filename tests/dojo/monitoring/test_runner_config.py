from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_runner_to_run_config_propagates_monitoring_config() -> None:
    source = (ROOT / "src" / "dojo" / "main_runner_job_array.py").read_text(encoding="utf-8")

    assert "monitoring=runner_cfg.monitoring" in source
