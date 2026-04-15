# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

METADATA_FILE = "metadata.json"
PHASE_EVENTS_FILE = "phase_events.jsonl"
PROCESS_SAMPLES_FILE = "process_samples.jsonl"
GPU_SAMPLES_FILE = "gpu_samples.jsonl"
GPU_PROCESS_SAMPLES_FILE = "gpu_process_samples.jsonl"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.md"
PLOTS_DIR = "plots"


def raw_jsonl_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "phase_events": output_dir / PHASE_EVENTS_FILE,
        "process_samples": output_dir / PROCESS_SAMPLES_FILE,
        "gpu_samples": output_dir / GPU_SAMPLES_FILE,
        "gpu_process_samples": output_dir / GPU_PROCESS_SAMPLES_FILE,
    }
