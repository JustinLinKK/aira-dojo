# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import shutil
import subprocess
import time
from io import StringIO
from typing import Any

log = logging.getLogger(__name__)

GPU_QUERY_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
]
GPU_PROCESS_QUERY_FIELDS = ["pid", "process_name", "used_memory"]
DRIVER_QUERY_FIELDS = ["driver_version", "cuda_version"]


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _parse_optional_float(value: str) -> float | None:
    clean = value.strip()
    if clean in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def _parse_optional_int(value: str) -> int | None:
    parsed = _parse_optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _rows(output: str) -> list[list[str]]:
    if not output.strip():
        return []
    reader = csv.reader(StringIO(output))
    return [[cell.strip() for cell in row] for row in reader if row]


def parse_gpu_samples(output: str, ts: float | None = None) -> list[dict[str, Any]]:
    sample_ts = time.time() if ts is None else ts
    samples: list[dict[str, Any]] = []
    for row in _rows(output):
        if len(row) < len(GPU_QUERY_FIELDS):
            continue
        samples.append(
            {
                "ts": sample_ts,
                "gpu_index": _parse_optional_int(row[0]),
                "name": row[1],
                "util_percent": _parse_optional_float(row[2]),
                "mem_used_mb": _parse_optional_float(row[3]),
                "mem_total_mb": _parse_optional_float(row[4]),
                "power_w": _parse_optional_float(row[5]),
                "temperature_c": _parse_optional_float(row[6]),
            }
        )
    return samples


def parse_gpu_process_samples(output: str, ts: float | None = None) -> list[dict[str, Any]]:
    sample_ts = time.time() if ts is None else ts
    samples: list[dict[str, Any]] = []
    for row in _rows(output):
        if len(row) < len(GPU_PROCESS_QUERY_FIELDS):
            continue
        pid = _parse_optional_int(row[0])
        if pid is None:
            continue
        samples.append(
            {
                "ts": sample_ts,
                "pid": pid,
                "process_name": row[1],
                "used_memory_mb": _parse_optional_float(row[2]),
            }
        )
    return samples


def _run_nvidia_smi(query_type: str, fields: list[str], timeout_sec: float = 3.0) -> str:
    if not nvidia_smi_available():
        return ""
    command = [
        "nvidia-smi",
        f"--query-{query_type}={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=timeout_sec)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.debug("nvidia-smi query failed: %s", exc)
        return ""


def query_gpu_samples(ts: float | None = None) -> list[dict[str, Any]]:
    return parse_gpu_samples(_run_nvidia_smi("gpu", GPU_QUERY_FIELDS), ts=ts)


def query_gpu_process_samples(ts: float | None = None) -> list[dict[str, Any]]:
    return parse_gpu_process_samples(_run_nvidia_smi("compute-apps", GPU_PROCESS_QUERY_FIELDS), ts=ts)


def query_driver_info() -> dict[str, str | None]:
    output = _run_nvidia_smi("gpu", DRIVER_QUERY_FIELDS)
    rows = _rows(output)
    if not rows:
        return {"driver_version": None, "cuda_version": None}
    row = rows[0]
    return {
        "driver_version": row[0] if len(row) > 0 and row[0] != "N/A" else None,
        "cuda_version": row[1] if len(row) > 1 and row[1] != "N/A" else None,
    }
