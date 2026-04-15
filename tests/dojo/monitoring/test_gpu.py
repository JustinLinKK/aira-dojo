from dojo.monitoring.gpu import parse_gpu_process_samples, parse_gpu_samples, query_gpu_samples


def test_parse_gpu_samples_handles_normal_and_na_rows() -> None:
    output = "\n".join(
        [
            "0, NVIDIA RTX, 35, 12000, 32607, 180.5, 62",
            "1, NVIDIA RTX, N/A, 0, 32607, N/A, N/A",
        ]
    )

    samples = parse_gpu_samples(output, ts=123.0)

    assert samples[0]["gpu_index"] == 0
    assert samples[0]["util_percent"] == 35.0
    assert samples[0]["power_w"] == 180.5
    assert samples[1]["util_percent"] is None
    assert samples[1]["power_w"] is None


def test_parse_gpu_process_samples_handles_blank_and_normal_rows() -> None:
    assert parse_gpu_process_samples("", ts=123.0) == []

    samples = parse_gpu_process_samples("1234, python, 2048\n", ts=123.0)

    assert samples == [{"ts": 123.0, "pid": 1234, "process_name": "python", "used_memory_mb": 2048.0}]


def test_query_gpu_samples_missing_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr("dojo.monitoring.gpu.nvidia_smi_available", lambda: False)

    assert query_gpu_samples(ts=123.0) == []
