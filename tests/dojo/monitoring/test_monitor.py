import json
import time

from dojo.config_dataclasses.monitoring import MonitoringConfig
from dojo.monitoring.monitor import ResourceMonitor
from dojo.monitoring.phases import get_active_monitor
from dojo.monitoring.report import build_phase_spans


def _fast_gpu_sample(ts: float | None = None) -> list[dict]:
    return [
        {
            "ts": time.time() if ts is None else ts,
            "gpu_index": 0,
            "name": "Test GPU",
            "util_percent": 0.0,
            "mem_used_mb": 100.0,
            "mem_total_mb": 1000.0,
            "power_w": 10.0,
            "temperature_c": 30.0,
        }
    ]


def test_monitor_starts_stops_and_writes_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("dojo.monitoring.monitor.query_gpu_samples", _fast_gpu_sample)
    monkeypatch.setattr("dojo.monitoring.monitor.query_gpu_process_samples", lambda ts=None: [])
    monkeypatch.setattr(
        "dojo.monitoring.monitor.query_driver_info",
        lambda: {"driver_version": "1", "cuda_version": "2"},
    )

    cfg = MonitoringConfig(
        baseline_seconds=0.0,
        sample_interval_sec=0.05,
        gpu_query_interval_sec=0.05,
        write_report=False,
    )
    monitor = ResourceMonitor(cfg, output_dir=tmp_path)

    monitor.start()
    assert get_active_monitor() is monitor
    with monitor.phase("outer"):
        with monitor.phase("inner", step=1):
            time.sleep(0.12)
    monitor.stop()

    output_dir = tmp_path / "resource_monitor"
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "phase_events.jsonl").exists()
    assert (output_dir / "process_samples.jsonl").exists()
    assert (output_dir / "gpu_samples.jsonl").exists()
    assert get_active_monitor() is None

    events = [json.loads(line) for line in (output_dir / "phase_events.jsonl").read_text().splitlines()]
    spans = build_phase_spans(events)
    assert [span["phase"] for span in spans] == ["outer", "inner"]
    assert spans[1]["step"] == 1


def test_disabled_monitor_is_noop(tmp_path) -> None:
    cfg = MonitoringConfig(enabled=False)
    monitor = ResourceMonitor.from_config(cfg, output_dir=tmp_path)

    monitor.start()
    with monitor.phase("ignored"):
        pass
    monitor.stop()
    monitor.write_report()

    assert not (tmp_path / "resource_monitor").exists()
