import json

from dojo.monitoring.report import generate_report


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_report_generation_from_synthetic_samples(tmp_path) -> None:
    monitor_dir = tmp_path / "resource_monitor"
    monitor_dir.mkdir()
    metadata = {
        "command": ["dojo"],
        "git_sha": "abc123",
        "monitoring_config": {
            "idle_gpu_util_threshold": 10,
            "gpu_memory_idle_threshold_mb": 1024,
        },
        "system": {"logical_cpu_count": 4},
        "gpu": {"devices": [{"gpu_index": 0, "name": "Test GPU", "mem_total_mb": 16000}]},
        "baseline": {"gpu": {"0": {"mem_used_mb_mean": 500, "power_w_mean": 20}}},
    }
    (monitor_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _write_jsonl(
        monitor_dir / "phase_events.jsonl",
        [
            {"ts": 0.0, "event": "begin", "phase": "framework_total", "depth": 0},
            {"ts": 1.0, "event": "begin", "phase": "generated_solution_execute", "depth": 1, "step": 1},
            {"ts": 5.0, "event": "end", "phase": "generated_solution_execute", "depth": 1, "step": 1},
            {"ts": 6.0, "event": "end", "phase": "framework_total", "depth": 0},
        ],
    )
    _write_jsonl(
        monitor_dir / "process_samples.jsonl",
        [
            {
                "ts": 1.0,
                "cpu_percent_total": 100.0,
                "rss_mb_total": 1000,
                "read_bytes_total": 0,
                "write_bytes_total": 0,
            },
            {
                "ts": 2.0,
                "cpu_percent_total": 200.0,
                "rss_mb_total": 1200,
                "read_bytes_total": 100,
                "write_bytes_total": 100,
            },
            {
                "ts": 3.0,
                "cpu_percent_total": 100.0,
                "rss_mb_total": 1100,
                "read_bytes_total": 200,
                "write_bytes_total": 200,
            },
            {
                "ts": 4.0,
                "cpu_percent_total": 100.0,
                "rss_mb_total": 900,
                "read_bytes_total": 300,
                "write_bytes_total": 300,
            },
            {
                "ts": 5.0,
                "cpu_percent_total": 100.0,
                "rss_mb_total": 800,
                "read_bytes_total": 400,
                "write_bytes_total": 400,
            },
        ],
    )
    _write_jsonl(
        monitor_dir / "gpu_samples.jsonl",
        [
            {
                "ts": 1.0,
                "gpu_index": 0,
                "util_percent": 0.0,
                "mem_used_mb": 12000,
                "mem_total_mb": 16000,
                "power_w": 120,
            },
            {
                "ts": 2.0,
                "gpu_index": 0,
                "util_percent": 0.0,
                "mem_used_mb": 12000,
                "mem_total_mb": 16000,
                "power_w": 120,
            },
            {
                "ts": 3.0,
                "gpu_index": 0,
                "util_percent": 50.0,
                "mem_used_mb": 14000,
                "mem_total_mb": 16000,
                "power_w": 180,
            },
            {
                "ts": 4.0,
                "gpu_index": 0,
                "util_percent": 0.0,
                "mem_used_mb": 12000,
                "mem_total_mb": 16000,
                "power_w": 120,
            },
            {
                "ts": 5.0,
                "gpu_index": 0,
                "util_percent": 0.0,
                "mem_used_mb": 12000,
                "mem_total_mb": 16000,
                "power_w": 120,
            },
        ],
    )
    _write_jsonl(monitor_dir / "gpu_process_samples.jsonl", [])

    summary = generate_report(monitor_dir)

    assert summary["training"]["wall_seconds"] == 4.0
    assert summary["waste_indicators"]["gpu_idle_wall_time_sec"] > 0
    assert summary["waste_indicators"]["gpu_reserved_idle_memory_mb_sec"] > 0
    assert (monitor_dir / "summary.json").exists()
    assert (monitor_dir / "report.md").exists()
    assert (monitor_dir / "plots" / "gpu_util.png").exists()
