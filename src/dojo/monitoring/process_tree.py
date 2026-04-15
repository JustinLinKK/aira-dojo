# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
import time
from typing import Any, Iterable

import psutil


def _safe_process(pid: int) -> psutil.Process | None:
    try:
        return psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def _collect_processes(root_pid: int, extra_pids: Iterable[int] | None = None) -> dict[int, psutil.Process]:
    processes: dict[int, psutil.Process] = {}
    roots = [root_pid, *(extra_pids or [])]
    for pid in roots:
        proc = _safe_process(pid)
        if proc is None:
            continue
        processes[proc.pid] = proc
        try:
            for child in proc.children(recursive=True):
                processes[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def aggregate_process_tree(
    root_pid: int | None = None,
    extra_pids: Iterable[int] | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    sample_ts = time.time() if ts is None else ts
    root = os.getpid() if root_pid is None else root_pid
    processes = _collect_processes(root, extra_pids)

    cpu_percent_total = 0.0
    rss_mb_total = 0.0
    read_bytes_total = 0
    write_bytes_total = 0
    pids: list[int] = []

    for proc in processes.values():
        try:
            pids.append(proc.pid)
            cpu_percent_total += proc.cpu_percent(interval=None)
            rss_mb_total += proc.memory_info().rss / (1024 * 1024)
            try:
                io_counters = proc.io_counters()
                read_bytes_total += int(io_counters.read_bytes)
                write_bytes_total += int(io_counters.write_bytes)
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        "ts": sample_ts,
        "root_pid": root,
        "pids": sorted(set(pids)),
        "cpu_percent_total": cpu_percent_total,
        "rss_mb_total": rss_mb_total,
        "read_bytes_total": read_bytes_total,
        "write_bytes_total": write_bytes_total,
    }


def system_metadata() -> dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor() or None,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "cpu_freq_mhz": cpu_freq.current if cpu_freq is not None else None,
        "ram_total_mb": virtual_memory.total / (1024 * 1024),
    }
