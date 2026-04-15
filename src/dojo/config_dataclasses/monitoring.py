# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from aira_core.config.base import BaseConfig


@dataclass
class MonitoringConfig(BaseConfig):
    enabled: bool = field(
        default=True,
        metadata={"help": "Whether to collect local CPU/GPU resource monitoring samples.", "exclude_from_hash": True},
    )
    sample_interval_sec: float = field(
        default=1.0,
        metadata={"help": "Interval between process-tree samples.", "exclude_from_hash": True},
    )
    baseline_seconds: float = field(
        default=10.0,
        metadata={
            "help": "Seconds to sample baseline resources before framework_total starts.",
            "exclude_from_hash": True,
        },
    )
    gpu_query_interval_sec: float = field(
        default=1.0,
        metadata={"help": "Interval between nvidia-smi GPU samples.", "exclude_from_hash": True},
    )
    track_process_tree: bool = field(
        default=True,
        metadata={"help": "Whether to aggregate root/child process CPU, memory, and I/O.", "exclude_from_hash": True},
    )
    track_gpu_processes: bool = field(
        default=True,
        metadata={"help": "Whether to sample nvidia-smi compute-app process rows.", "exclude_from_hash": True},
    )
    write_raw_samples: bool = field(
        default=True,
        metadata={"help": "Whether to write raw JSONL sample files.", "exclude_from_hash": True},
    )
    write_report: bool = field(
        default=True,
        metadata={
            "help": "Whether to write summary.json, report.md, and plots at shutdown.",
            "exclude_from_hash": True,
        },
    )
    idle_gpu_util_threshold: float = field(
        default=10.0,
        metadata={
            "help": "GPU utilization percentage below which training samples are considered idle.",
            "exclude_from_hash": True,
        },
    )
    idle_cpu_util_threshold: float = field(
        default=20.0,
        metadata={
            "help": "CPU utilization percentage below which process-tree samples are considered idle.",
            "exclude_from_hash": True,
        },
    )
    gpu_memory_idle_threshold_mb: float = field(
        default=1024.0,
        metadata={
            "help": "Reserved GPU memory threshold for suspicious idle-memory windows.",
            "exclude_from_hash": True,
        },
    )
    output_subdir: str = field(
        default="resource_monitor",
        metadata={"help": "Subdirectory under logger.output_dir for monitoring artifacts.", "exclude_from_hash": True},
    )

    def validate(self) -> None:
        super().validate()
        if self.sample_interval_sec <= 0:
            raise ValueError("monitoring.sample_interval_sec must be positive.")
        if self.baseline_seconds < 0:
            raise ValueError("monitoring.baseline_seconds must be non-negative.")
        if self.gpu_query_interval_sec <= 0:
            raise ValueError("monitoring.gpu_query_interval_sec must be positive.")
