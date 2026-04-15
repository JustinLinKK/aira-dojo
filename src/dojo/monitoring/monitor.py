# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from dojo.config_dataclasses.monitoring import MonitoringConfig
from dojo.monitoring.config import (
    GPU_PROCESS_SAMPLES_FILE,
    GPU_SAMPLES_FILE,
    METADATA_FILE,
    PHASE_EVENTS_FILE,
    PROCESS_SAMPLES_FILE,
)
from dojo.monitoring.gpu import query_driver_info, query_gpu_process_samples, query_gpu_samples
from dojo.monitoring.phases import set_active_monitor
from dojo.monitoring.process_tree import aggregate_process_tree, system_metadata

log = logging.getLogger(__name__)


class NullResourceMonitor:
    output_dir: Path | None = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def write_report(self) -> None:
        return None

    @contextmanager
    def phase(self, phase: str, **metadata: Any) -> Iterator[None]:
        del phase, metadata
        with nullcontext():
            yield

    def register_process(self, pid: int, label: str | None = None) -> None:
        del pid, label

    def unregister_process(self, pid: int) -> None:
        del pid


class ResourceMonitor:
    def __init__(self, cfg: MonitoringConfig, output_dir: str | Path, root_pid: int | None = None) -> None:
        self.cfg = cfg
        self.root_pid = os.getpid() if root_pid is None else root_pid
        self.output_dir = Path(output_dir) / cfg.output_subdir
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._registered_pids: dict[int, str | None] = {}
        self._registered_pids_lock = threading.Lock()
        self._phase_local = threading.local()
        self._last_gpu_query_ts = 0.0
        self._baseline_process_samples: list[dict[str, Any]] = []
        self._baseline_gpu_samples: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls,
        cfg: MonitoringConfig | None,
        output_dir: str | Path,
    ) -> "ResourceMonitor | NullResourceMonitor":
        if cfg is None or not cfg.enabled:
            return NullResourceMonitor()
        return cls(cfg=cfg, output_dir=output_dir)

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_output_files()
        set_active_monitor(self)
        self._metadata = self._build_metadata()
        self._write_metadata()

        if self.cfg.baseline_seconds > 0:
            with self.phase("monitor_baseline"):
                self._collect_baseline()
            self._metadata["baseline"] = self._summarize_baseline()
            self._write_metadata()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_sampler, name="dojo-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.cfg.sample_interval_sec * 2))
            self._thread = None
        self._sample_once()
        set_active_monitor(None)

    def write_report(self) -> None:
        if not self.cfg.write_report:
            return
        try:
            from dojo.monitoring.report import generate_report

            generate_report(self.output_dir)
        except Exception as exc:
            log.warning("Failed to write resource monitor report: %s", exc)

    @contextmanager
    def phase(self, phase: str, **metadata: Any) -> Iterator[None]:
        stack = self._phase_stack()
        depth = len(stack)
        event_metadata = {key: value for key, value in metadata.items() if value is not None}
        self._write_phase_event("begin", phase, depth=depth, metadata=event_metadata)
        stack.append(phase)
        try:
            yield
        finally:
            if stack:
                stack.pop()
            self._write_phase_event("end", phase, depth=depth, metadata=event_metadata)

    def register_process(self, pid: int, label: str | None = None) -> None:
        with self._registered_pids_lock:
            self._registered_pids[pid] = label

    def unregister_process(self, pid: int) -> None:
        with self._registered_pids_lock:
            self._registered_pids.pop(pid, None)

    def _phase_stack(self) -> list[str]:
        stack = getattr(self._phase_local, "stack", None)
        if stack is None:
            stack = []
            self._phase_local.stack = stack
        return stack

    def _registered_pid_snapshot(self) -> list[int]:
        with self._registered_pids_lock:
            return list(self._registered_pids)

    def _initialize_output_files(self) -> None:
        filenames = [PHASE_EVENTS_FILE]
        if self.cfg.write_raw_samples:
            filenames.extend([PROCESS_SAMPLES_FILE, GPU_SAMPLES_FILE, GPU_PROCESS_SAMPLES_FILE])
        for filename in filenames:
            (self.output_dir / filename).touch(exist_ok=True)

    def _collect_baseline(self) -> None:
        deadline = time.time() + self.cfg.baseline_seconds
        while True:
            self._sample_once()
            if time.time() >= deadline:
                break
            time.sleep(min(self.cfg.sample_interval_sec, max(0.0, deadline - time.time())))

    def _run_sampler(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            self._sample_once()
            elapsed = time.time() - started
            wait_for = max(0.05, self.cfg.sample_interval_sec - elapsed)
            self._stop_event.wait(wait_for)

    def _sample_once(self) -> None:
        ts = time.time()
        if self.cfg.track_process_tree:
            process_sample = aggregate_process_tree(
                root_pid=self.root_pid,
                extra_pids=self._registered_pid_snapshot(),
                ts=ts,
            )
            if self._in_baseline_phase():
                self._baseline_process_samples.append(process_sample)
            if self.cfg.write_raw_samples:
                self._write_jsonl(PROCESS_SAMPLES_FILE, process_sample)

        should_query_gpu = ts - self._last_gpu_query_ts >= self.cfg.gpu_query_interval_sec
        if should_query_gpu:
            self._last_gpu_query_ts = ts
            gpu_samples = query_gpu_samples(ts=ts)
            if self._in_baseline_phase():
                self._baseline_gpu_samples.extend(gpu_samples)
            if self.cfg.write_raw_samples:
                for gpu_sample in gpu_samples:
                    self._write_jsonl(GPU_SAMPLES_FILE, gpu_sample)

            if self.cfg.track_gpu_processes:
                gpu_process_samples = query_gpu_process_samples(ts=ts)
                if self.cfg.write_raw_samples:
                    for gpu_process_sample in gpu_process_samples:
                        self._write_jsonl(GPU_PROCESS_SAMPLES_FILE, gpu_process_sample)

    def _in_baseline_phase(self) -> bool:
        return "monitor_baseline" in self._phase_stack()

    def _write_phase_event(self, event: str, phase: str, depth: int, metadata: dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "event": event,
            "phase": phase,
            "depth": depth,
            **metadata,
        }
        self._write_jsonl(PHASE_EVENTS_FILE, row)

    def _write_jsonl(self, filename: str, row: dict[str, Any]) -> None:
        path = self.output_dir / filename
        with self._write_lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")

    def _write_metadata(self) -> None:
        with self._write_lock:
            with (self.output_dir / METADATA_FILE).open("w", encoding="utf-8") as file:
                json.dump(_jsonable(self._metadata), file, indent=2, sort_keys=True)

    def _build_metadata(self) -> dict[str, Any]:
        gpu_samples = query_gpu_samples()
        metadata = {
            "command": sys.argv,
            "cwd": os.getcwd(),
            "root_pid": self.root_pid,
            "git_sha": _git_sha(),
            "monitoring_config": asdict(self.cfg) if is_dataclass(self.cfg) else dict(self.cfg),
            "sample_interval_sec": self.cfg.sample_interval_sec,
            "gpu_query_interval_sec": self.cfg.gpu_query_interval_sec,
            "system": system_metadata(),
            "gpu": {
                "driver": query_driver_info(),
                "devices": [
                    {
                        "gpu_index": sample.get("gpu_index"),
                        "name": sample.get("name"),
                        "mem_total_mb": sample.get("mem_total_mb"),
                    }
                    for sample in gpu_samples
                ],
            },
            "python": {
                "executable": sys.executable,
                "version": sys.version,
                "torch_version": _module_version("torch"),
            },
        }
        return metadata

    def _summarize_baseline(self) -> dict[str, Any]:
        process_cpu = [float(sample["cpu_percent_total"]) for sample in self._baseline_process_samples]
        process_rss = [float(sample["rss_mb_total"]) for sample in self._baseline_process_samples]
        gpu_by_index: dict[str, list[dict[str, Any]]] = {}
        for sample in self._baseline_gpu_samples:
            gpu_by_index.setdefault(str(sample.get("gpu_index")), []).append(sample)

        return {
            "duration_sec": self.cfg.baseline_seconds,
            "process": {
                "cpu_percent_total_mean": _mean(process_cpu),
                "rss_mb_total_mean": _mean(process_rss),
            },
            "gpu": {
                gpu_index: {
                    "util_percent_mean": _mean([sample.get("util_percent") for sample in samples]),
                    "mem_used_mb_mean": _mean([sample.get("mem_used_mb") for sample in samples]),
                    "power_w_mean": _mean([sample.get("power_w") for sample in samples]),
                }
                for gpu_index, samples in gpu_by_index.items()
            },
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", None))
    except Exception:
        return None
