# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from dojo.monitoring.config import (
    GPU_PROCESS_SAMPLES_FILE,
    GPU_SAMPLES_FILE,
    METADATA_FILE,
    PHASE_EVENTS_FILE,
    PLOTS_DIR,
    PROCESS_SAMPLES_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
)


def generate_report(resource_monitor_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(resource_monitor_dir)
    metadata = _read_json(output_dir / METADATA_FILE, default={})
    phase_events = _read_jsonl(output_dir / PHASE_EVENTS_FILE)
    process_samples = _read_jsonl(output_dir / PROCESS_SAMPLES_FILE)
    gpu_samples = _read_jsonl(output_dir / GPU_SAMPLES_FILE)
    gpu_process_samples = _read_jsonl(output_dir / GPU_PROCESS_SAMPLES_FILE)

    spans = build_phase_spans(phase_events)
    summary = build_summary(metadata, spans, process_samples, gpu_samples, gpu_process_samples)

    plots_dir = output_dir / PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_plots(plots_dir, process_samples, gpu_samples, spans)

    with (output_dir / SUMMARY_FILE).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    (output_dir / REPORT_FILE).write_text(_markdown_report(summary, output_dir), encoding="utf-8")
    return summary


def build_phase_spans(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: float(row.get("ts", 0.0))):
        if event.get("event") == "begin":
            stack.append(event)
            continue
        if event.get("event") != "end":
            continue

        matching_index = None
        for index in range(len(stack) - 1, -1, -1):
            if stack[index].get("phase") == event.get("phase"):
                matching_index = index
                break
        if matching_index is None:
            continue
        begin = stack.pop(matching_index)
        start_ts = float(begin.get("ts", 0.0))
        end_ts = float(event.get("ts", start_ts))
        if end_ts < start_ts:
            continue
        metadata = {
            key: value
            for key, value in begin.items()
            if key not in {"ts", "event", "phase", "depth"}
        }
        spans.append(
            {
                "phase": begin.get("phase"),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_sec": end_ts - start_ts,
                "depth": int(begin.get("depth", 0)),
                **metadata,
            }
        )
    return sorted(spans, key=lambda row: (float(row["start_ts"]), int(row.get("depth", 0))))


def build_summary(
    metadata: dict[str, Any],
    spans: list[dict[str, Any]],
    process_samples: list[dict[str, Any]],
    gpu_samples: list[dict[str, Any]],
    gpu_process_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = metadata.get("monitoring_config", {})
    idle_gpu_threshold = float(cfg.get("idle_gpu_util_threshold", 10.0))
    idle_memory_threshold = float(cfg.get("gpu_memory_idle_threshold_mb", 1024.0))
    logical_cpu_count = int(metadata.get("system", {}).get("logical_cpu_count") or 1)

    durations = _phase_durations(spans)
    training_spans = [span for span in spans if span.get("phase") == "generated_solution_execute"]
    training_wall_seconds = sum(float(span["duration_sec"]) for span in training_spans)

    process_training = _timeline_for_spans(process_samples, training_spans)
    gpu_training = _timeline_for_spans(_aggregate_gpu_samples(gpu_samples), training_spans)
    baseline_gpu_mem_mb = _baseline_gpu_value(metadata, "mem_used_mb_mean")
    baseline_gpu_power_w = _baseline_gpu_value(metadata, "power_w_mean")

    process_core_seconds = sum(
        max(0.0, float(sample.get("cpu_percent_total") or 0.0)) / 100.0 * float(sample.get("_dt", 0.0))
        for sample in process_training
    )
    cpu_unused_core_seconds = max(0.0, training_wall_seconds * logical_cpu_count - process_core_seconds)

    gpu_idle_wall_time_sec = sum(
        float(sample.get("_dt", 0.0))
        for sample in gpu_training
        if _lt(sample.get("util_percent"), idle_gpu_threshold)
    )
    gpu_reserved_idle_memory_mb_sec = sum(
        max(0.0, float(sample.get("mem_used_mb") or 0.0) - baseline_gpu_mem_mb) * float(sample.get("_dt", 0.0))
        for sample in gpu_training
        if _lt(sample.get("util_percent"), idle_gpu_threshold)
    )
    gpu_low_util_power_wh = sum(
        max(0.0, float(sample.get("power_w") or 0.0) - baseline_gpu_power_w) * float(sample.get("_dt", 0.0)) / 3600.0
        for sample in gpu_training
        if _lt(sample.get("util_percent"), idle_gpu_threshold)
    )
    cpu_bottleneck_signal_sec = _overlap_signal_seconds(
        process_training,
        gpu_training,
        lambda process, gpu: _lt(gpu.get("util_percent"), idle_gpu_threshold)
        and float(process.get("cpu_percent_total") or 0.0) > logical_cpu_count * 80.0,
    )
    high_io_threshold = _io_activity_threshold(process_samples)
    data_or_io_bottleneck_signal_sec = _overlap_signal_seconds(
        process_training,
        gpu_training,
        lambda process, gpu: _lt(gpu.get("util_percent"), idle_gpu_threshold)
        and _io_delta_bytes(process) >= high_io_threshold,
    )

    peak_gpu_mem_used_mb = _max(sample.get("mem_used_mb") for sample in gpu_training)
    peak_gpu_mem_total_mb = _max(sample.get("mem_total_mb") for sample in gpu_training)
    gpu_memory_headroom_mb = (
        max(0.0, peak_gpu_mem_total_mb - peak_gpu_mem_used_mb)
        if peak_gpu_mem_total_mb is not None and peak_gpu_mem_used_mb is not None
        else None
    )

    return {
        "environment": {
            "command": metadata.get("command"),
            "git_sha": metadata.get("git_sha"),
            "system": metadata.get("system", {}),
            "gpu": metadata.get("gpu", {}),
        },
        "phase_durations_sec": durations,
        "training": {
            "span_count": len(training_spans),
            "wall_seconds": training_wall_seconds,
            "cpu_core_seconds": process_core_seconds,
            "peak_rss_mb": _max(sample.get("rss_mb_total") for sample in process_training),
            "peak_gpu_mem_used_mb": peak_gpu_mem_used_mb,
            "peak_gpu_mem_total_mb": peak_gpu_mem_total_mb,
        },
        "waste_indicators": {
            "gpu_idle_wall_time_sec": gpu_idle_wall_time_sec,
            "gpu_reserved_idle_memory_mb_sec": gpu_reserved_idle_memory_mb_sec,
            "gpu_low_util_power_wh": gpu_low_util_power_wh,
            "cpu_unused_core_seconds": cpu_unused_core_seconds,
            "cpu_bottleneck_signal_sec": cpu_bottleneck_signal_sec,
            "data_or_io_bottleneck_signal_sec": data_or_io_bottleneck_signal_sec,
            "gpu_memory_headroom_mb": gpu_memory_headroom_mb,
        },
        "suspicious_windows": _suspicious_windows(gpu_training, idle_gpu_threshold, idle_memory_threshold),
        "sample_counts": {
            "process_samples": len(process_samples),
            "gpu_samples": len(gpu_samples),
            "gpu_process_samples": len(gpu_process_samples),
            "phase_events": len(spans),
        },
        "raw_files": {
            "metadata": METADATA_FILE,
            "phase_events": PHASE_EVENTS_FILE,
            "process_samples": PROCESS_SAMPLES_FILE,
            "gpu_samples": GPU_SAMPLES_FILE,
            "gpu_process_samples": GPU_PROCESS_SAMPLES_FILE,
        },
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _phase_durations(spans: list[dict[str, Any]]) -> dict[str, float]:
    durations: dict[str, float] = defaultdict(float)
    for span in spans:
        durations[str(span.get("phase"))] += float(span.get("duration_sec", 0.0))
    return dict(sorted(durations.items(), key=lambda item: item[0]))


def _aggregate_gpu_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[float(sample.get("ts", 0.0))].append(sample)

    timeline: list[dict[str, Any]] = []
    for ts, rows in grouped.items():
        timeline.append(
            {
                "ts": ts,
                "util_percent": _max(row.get("util_percent") for row in rows),
                "mem_used_mb": sum(float(row.get("mem_used_mb") or 0.0) for row in rows),
                "mem_total_mb": sum(float(row.get("mem_total_mb") or 0.0) for row in rows),
                "power_w": sum(float(row.get("power_w") or 0.0) for row in rows),
                "temperature_c": _max(row.get("temperature_c") for row in rows),
            }
        )
    return sorted(timeline, key=lambda row: float(row["ts"]))


def _timeline_for_spans(samples: list[dict[str, Any]], spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not samples or not spans:
        return []
    sorted_samples = sorted(samples, key=lambda row: float(row.get("ts", 0.0)))
    timeline: list[dict[str, Any]] = []
    for index, sample in enumerate(sorted_samples):
        ts = float(sample.get("ts", 0.0))
        span = _span_for_ts(spans, ts)
        if span is None:
            continue
        next_ts = (
            min(float(sorted_samples[index + 1].get("ts", ts)), float(span["end_ts"]))
            if index + 1 < len(sorted_samples)
            else float(span["end_ts"])
        )
        dt = max(0.0, next_ts - ts)
        enriched = dict(sample)
        enriched["_dt"] = dt
        enriched["_phase"] = span.get("phase")
        timeline.append(enriched)
    return timeline


def _span_for_ts(spans: list[dict[str, Any]], ts: float) -> dict[str, Any] | None:
    containing = [
        span
        for span in spans
        if float(span.get("start_ts", 0.0)) <= ts <= float(span.get("end_ts", 0.0))
    ]
    if not containing:
        return None
    return max(containing, key=lambda span: int(span.get("depth", 0)))


def _baseline_gpu_value(metadata: dict[str, Any], key: str) -> float:
    baseline_gpu = metadata.get("baseline", {}).get("gpu", {})
    values = [float(gpu[key]) for gpu in baseline_gpu.values() if isinstance(gpu.get(key), (int, float))]
    return sum(values)


def _overlap_signal_seconds(
    process_timeline: list[dict[str, Any]],
    gpu_timeline: list[dict[str, Any]],
    predicate: Any,
) -> float:
    total = 0.0
    for process_sample in process_timeline:
        process_ts = float(process_sample.get("ts", 0.0))
        gpu_sample = _nearest_at_or_before(gpu_timeline, process_ts)
        if gpu_sample is not None and predicate(process_sample, gpu_sample):
            total += float(process_sample.get("_dt", 0.0))
    return total


def _nearest_at_or_before(samples: list[dict[str, Any]], ts: float) -> dict[str, Any] | None:
    candidate = None
    for sample in samples:
        if float(sample.get("ts", 0.0)) <= ts:
            candidate = sample
        else:
            break
    return candidate


def _io_activity_threshold(samples: list[dict[str, Any]]) -> float:
    deltas = [_io_delta_bytes(sample) for sample in samples]
    if not deltas:
        return math.inf
    sorted_deltas = sorted(deltas)
    index = min(len(sorted_deltas) - 1, int(len(sorted_deltas) * 0.9))
    return max(1.0, float(sorted_deltas[index]))


def _io_delta_bytes(sample: dict[str, Any]) -> float:
    return float(sample.get("read_bytes_total") or 0.0) + float(sample.get("write_bytes_total") or 0.0)


def _suspicious_windows(
    gpu_training: list[dict[str, Any]],
    idle_gpu_threshold: float,
    idle_memory_threshold_mb: float,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for sample in gpu_training:
        idle_high_mem = _lt(sample.get("util_percent"), idle_gpu_threshold) and float(
            sample.get("mem_used_mb") or 0.0
        ) >= idle_memory_threshold_mb
        if idle_high_mem and current is None:
            current = {
                "start_ts": sample.get("ts"),
                "end_ts": float(sample.get("ts", 0.0)) + float(sample.get("_dt", 0.0)),
                "duration_sec": float(sample.get("_dt", 0.0)),
                "peak_mem_used_mb": sample.get("mem_used_mb"),
                "max_util_percent": sample.get("util_percent"),
            }
        elif idle_high_mem and current is not None:
            current["end_ts"] = float(sample.get("ts", 0.0)) + float(sample.get("_dt", 0.0))
            current["duration_sec"] = float(current.get("duration_sec") or 0.0) + float(sample.get("_dt", 0.0))
            current["peak_mem_used_mb"] = max(
                float(current.get("peak_mem_used_mb") or 0.0), float(sample.get("mem_used_mb") or 0.0)
            )
            current["max_util_percent"] = max(
                float(current.get("max_util_percent") or 0.0), float(sample.get("util_percent") or 0.0)
            )
        elif current is not None:
            windows.append(current)
            current = None
    if current is not None:
        windows.append(current)
    return sorted(windows, key=lambda row: float(row.get("duration_sec", 0.0)), reverse=True)[:10]


def _write_plots(
    plots_dir: Path,
    process_samples: list[dict[str, Any]],
    gpu_samples: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> None:
    del spans
    process_series = [
        ("cpu", process_samples, "cpu_percent_total", "CPU percent total"),
        ("rss", process_samples, "rss_mb_total", "RSS MB"),
    ]
    for filename, samples, key, title in process_series:
        _line_plot(plots_dir / f"{filename}.png", samples, key, title)

    gpu_timeline = _aggregate_gpu_samples(gpu_samples)
    gpu_series = [
        ("gpu_util", gpu_timeline, "util_percent", "GPU util percent"),
        ("gpu_memory", gpu_timeline, "mem_used_mb", "GPU memory used MB"),
        ("gpu_power", gpu_timeline, "power_w", "GPU power W"),
    ]
    for filename, samples, key, title in gpu_series:
        _line_plot(plots_dir / f"{filename}.png", samples, key, title)


def _line_plot(path: Path, samples: list[dict[str, Any]], key: str, title: str) -> None:
    plt.figure(figsize=(10, 3))
    if samples:
        start_ts = float(samples[0].get("ts", 0.0))
        xs = [float(sample.get("ts", 0.0)) - start_ts for sample in samples]
        ys = [sample.get(key) for sample in samples]
        plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel("seconds")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _markdown_report(summary: dict[str, Any], output_dir: Path) -> str:
    training = summary["training"]
    waste = summary["waste_indicators"]
    phases = summary["phase_durations_sec"]
    raw_files = summary["raw_files"]
    suspicious = summary["suspicious_windows"]

    lines = [
        "# Resource Monitor Report",
        "",
        "## Environment",
        f"- Git SHA: `{summary['environment'].get('git_sha')}`",
        f"- Command: `{summary['environment'].get('command')}`",
        f"- GPUs: `{summary['environment'].get('gpu', {}).get('devices', [])}`",
        "",
        "## Runtime By Phase",
        "| Phase | Seconds |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {phase} | {_fmt(seconds)} |" for phase, seconds in phases.items())
    lines.extend(
        [
            "",
            "## Generated Training Resource Summary",
            f"- Training spans: {training['span_count']}",
            f"- Training wall time: {_fmt(training['wall_seconds'])} sec",
            f"- Process CPU core-seconds: {_fmt(training['cpu_core_seconds'])}",
            f"- Peak RSS: {_fmt(training['peak_rss_mb'])} MB",
            f"- Peak GPU memory used: {_fmt(training['peak_gpu_mem_used_mb'])} MB",
            f"- GPU memory headroom: {_fmt(waste['gpu_memory_headroom_mb'])} MB",
            "",
            "## Waste Indicators",
            f"- GPU idle wall time during generated training: {_fmt(waste['gpu_idle_wall_time_sec'])} sec",
            "- Idle reserved GPU memory: "
            f"{_fmt(waste['gpu_reserved_idle_memory_mb_sec'])} MB-sec "
            "(VRAM allocation is only suspicious when compute utilization is near zero).",
            f"- Low-util GPU energy above baseline: {_fmt(waste['gpu_low_util_power_wh'])} Wh",
            f"- CPU unused core-seconds: {_fmt(waste['cpu_unused_core_seconds'])}",
            f"- CPU bottleneck signal: {_fmt(waste['cpu_bottleneck_signal_sec'])} sec",
            f"- Data/I/O bottleneck signal: {_fmt(waste['data_or_io_bottleneck_signal_sec'])} sec",
            "",
            "## Suspicious Windows",
        ]
    )
    if suspicious:
        lines.extend(
            "- GPU memory > threshold while GPU util is low for "
            f"{_fmt(window.get('duration_sec'))} sec, peak memory {_fmt(window.get('peak_mem_used_mb'))} MB"
            for window in suspicious
        )
    else:
        lines.append("- No sustained high-memory, low-utilization training windows found.")

    lines.extend(["", "## Raw Data"])
    lines.extend(f"- {name}: `{output_dir / filename}`" for name, filename in raw_files.items())
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _max(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return max(numeric)


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and float(value) < threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Dojo resource monitoring report.")
    parser.add_argument("resource_monitor_dir", type=Path)
    args = parser.parse_args()
    generate_report(args.resource_monitor_dir)


if __name__ == "__main__":
    main()
