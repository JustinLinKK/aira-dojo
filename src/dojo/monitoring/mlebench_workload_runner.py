# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dojo.config_dataclasses.monitoring import MonitoringConfig
from dojo.monitoring.monitor import ResourceMonitor
from dojo.utils.environment import get_mlebench_data_dir

TASK_NAME = "aerial-cactus-identification"
WORKLOAD_DIR = Path(__file__).resolve().parent / "workloads"
HISTOPATHOLOGIC_TASK_NAME = "histopathologic-cancer-detection"
BUILTIN_WORKLOADS = {
    "parallel": WORKLOAD_DIR / "aerial_cactus_parallel.py",
    "noop": WORKLOAD_DIR / "aerial_cactus_noop.py",
    "aerial_cactus_parallel": WORKLOAD_DIR / "aerial_cactus_parallel.py",
    "histopathologic_parallel": WORKLOAD_DIR / "histopathologic_parallel.py",
}


def workload_path(value: str, task_name: str = TASK_NAME) -> Path:
    if value == "parallel" and task_name == HISTOPATHOLOGIC_TASK_NAME:
        return BUILTIN_WORKLOADS["histopathologic_parallel"]
    if value in BUILTIN_WORKLOADS:
        return BUILTIN_WORKLOADS[value]
    return Path(value).expanduser().resolve()


def validate_prepared_task(data_dir: str | Path, task_name: str = TASK_NAME) -> dict[str, str]:
    task_root = Path(data_dir).expanduser().resolve() / task_name
    prepared_dir = task_root / "prepared"
    public_dir = prepared_dir / "public"
    private_dir = prepared_dir / "private"
    missing = [path for path in [public_dir, private_dir] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared MLE-Bench task data is missing. "
            f"Expected public/private dirs under {prepared_dir}. Missing: {', '.join(str(path) for path in missing)}"
        )
    sample_matches = sorted(public_dir.glob("**/sample_submission*.csv"))
    train_matches = sorted(public_dir.glob("**/train.csv"))
    if not train_matches:
        train_matches = sorted(public_dir.glob("**/train_labels.csv"))
    if not sample_matches:
        raise FileNotFoundError(f"No sample_submission*.csv found under {public_dir}")
    if not train_matches:
        raise FileNotFoundError(f"No train.csv or train_labels.csv found under {public_dir}")
    return {
        "task_root": str(task_root),
        "public_dir": str(public_dir),
        "private_dir": str(private_dir),
        "train_file": str(train_matches[0]),
        "train_csv": str(train_matches[0]),
        "sample_submission": str(sample_matches[0]),
    }


def build_workload_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        "MLEBENCH_WORKLOAD_PARALLEL_MODELS": str(args.parallel_models),
        "MLEBENCH_WORKLOAD_EPOCHS": str(args.epochs),
        "MLEBENCH_WORKLOAD_BATCH_SIZE": str(args.batch_size),
        "MLEBENCH_WORKLOAD_IMAGE_SIZE": str(args.image_size),
        "MLEBENCH_WORKLOAD_DATALOADER_WORKERS": str(args.dataloader_workers),
        "MLEBENCH_WORKLOAD_BASE_SEED": str(args.seed),
        "MLEBENCH_WORKLOAD_TRAIN_REPEAT": str(args.train_repeat),
        "AERIAL_CACTUS_PARALLEL_MODELS": str(args.parallel_models),
        "AERIAL_CACTUS_EPOCHS": str(args.epochs),
        "AERIAL_CACTUS_BATCH_SIZE": str(args.batch_size),
        "AERIAL_CACTUS_IMAGE_SIZE": str(args.image_size),
        "AERIAL_CACTUS_DATALOADER_WORKERS": str(args.dataloader_workers),
        "AERIAL_CACTUS_BASE_SEED": str(args.seed),
        "AERIAL_CACTUS_TRAIN_REPEAT": str(args.train_repeat),
        "HISTOPATHOLOGIC_PARALLEL_MODELS": str(args.parallel_models),
        "HISTOPATHOLOGIC_EPOCHS": str(args.epochs),
        "HISTOPATHOLOGIC_BATCH_SIZE": str(args.batch_size),
        "HISTOPATHOLOGIC_IMAGE_SIZE": str(args.image_size),
        "HISTOPATHOLOGIC_DATALOADER_WORKERS": str(args.dataloader_workers),
        "HISTOPATHOLOGIC_BASE_SEED": str(args.seed),
        "HISTOPATHOLOGIC_TRAIN_REPEAT": str(args.train_repeat),
    }


@contextmanager
def temporary_env(values: dict[str, str]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def summarize_eval_result(eval_result: dict[str, Any]) -> dict[str, Any]:
    from dojo.core.tasks.constants import EXECUTION_OUTPUT, TEST_FITNESS, VALID_SOLUTION, VALIDATION_FITNESS

    exec_output = eval_result.get(EXECUTION_OUTPUT)
    summary = {
        "success": False,
        "exit_code": None,
        "timed_out": None,
        "exec_time": None,
        "valid_solution": eval_result.get(VALID_SOLUTION, False),
        "validation_fitness": eval_result.get(VALIDATION_FITNESS),
        "test_fitness": eval_result.get(TEST_FITNESS),
        "stdout_tail": [],
    }
    if exec_output is not None:
        stdout = list(getattr(exec_output, "term_out", []) or [])
        summary.update(
            {
                "success": exec_output.exit_code == 0 and not exec_output.timed_out,
                "exit_code": exec_output.exit_code,
                "timed_out": exec_output.timed_out,
                "exec_time": exec_output.exec_time,
                "stdout_tail": stdout[-80:],
            }
        )
    return summary


def execute_monitored_workload(args: argparse.Namespace) -> dict[str, Any]:
    from dojo.config_dataclasses.interpreter.python import PythonInterpreterConfig
    from dojo.utils.logger import config_logger

    data_dir = Path(args.data_dir).expanduser().resolve()
    os.environ.setdefault("MLE_BENCH_DATA_DIR", str(data_dir))
    from dojo.core.interpreters.python import PythonInterpreter
    from dojo.config_dataclasses.task.mlebench import MLEBenchTaskConfig
    from dojo.tasks.mlebench.task import MLEBenchTask

    workload_file = workload_path(args.workload, args.task)
    if not workload_file.exists():
        raise FileNotFoundError(f"Workload code file not found: {workload_file}")

    prepared = validate_prepared_task(data_dir, args.task)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "workspace_agent").mkdir(parents=True, exist_ok=True)

    run_config = {
        "task": args.task,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "workload": str(workload_file),
        "workload_env": build_workload_env(args),
        "prepared": prepared,
    }
    (output_dir / "workload_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    task_config = MLEBenchTaskConfig(
        name=args.task,
        benchmark="mlebench",
        cache_dir=str(data_dir),
        public_dir=prepared["public_dir"],
        private_dir=prepared["private_dir"],
        data_dir=prepared["public_dir"],
        submission_fname="submission.csv",
        results_output_dir=str(output_dir / "results"),
    )
    interpreter_config = PythonInterpreterConfig(
        working_dir=str(output_dir / "workspace_agent"),
        timeout=int(args.timeout_hours * 60 * 60),
        use_symlinks=True,
    )
    monitor_config = MonitoringConfig(
        enabled=True,
        sample_interval_sec=args.sample_interval_sec,
        baseline_seconds=args.baseline_seconds,
        gpu_query_interval_sec=args.gpu_query_interval_sec,
        write_raw_samples=True,
        write_report=True,
    )

    config_logger(None)
    monitor = ResourceMonitor.from_config(monitor_config, output_dir=output_dir)
    task = None
    state = None
    eval_result: dict[str, Any] = {}
    monitor.start()
    try:
        with monitor.phase("framework_total"):
            with monitor.phase("preparation_check", task=args.task):
                validate_prepared_task(data_dir, args.task)
            with monitor.phase("task_instantiation", task=args.task):
                task = MLEBenchTask(task_config)
            with monitor.phase("interpreter_instantiation", interpreter="python"):
                solver_interpreter = PythonInterpreter(interpreter_config, data_dir=Path(task_config.data_dir))
            with monitor.phase("task_preparation", task=args.task):
                state, _ = task.prepare(solver_interpreter=solver_interpreter, eval_interpreter=None)
            with monitor.phase("fixed_solution_step", workload=str(workload_file)):
                with temporary_env(build_workload_env(args)):
                    _, eval_result = task.step_task(state, workload_file.read_text(encoding="utf-8"))
            summary = summarize_eval_result(eval_result)
            (output_dir / "workload_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
    finally:
        with monitor.phase("cleanup"):
            if task is not None and state is not None:
                task.close(state)
        monitor.stop()
        monitor.write_report()


def default_output_dir(task_name: str, workload: str, parallel_models: int, seed: int) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "mlebench_workloads" / f"{task_name}_{workload}_p{parallel_models}_seed{seed}_{stamp}"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a monitored deterministic MLE-Bench workload.")
    parser.add_argument("--task", default=TASK_NAME)
    parser.add_argument("--data-dir", default=None, help="Defaults to MLE_BENCH_DATA_DIR.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--workload",
        default="parallel",
        help="Built-ins: parallel, noop, aerial_cactus_parallel, histopathologic_parallel; or path to a Python file.",
    )
    parser.add_argument("--check-only", action="store_true", help="Only verify prepared task data and print paths.")
    parser.add_argument("--parallel-models", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--train-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--timeout-hours", type=float, default=4.0)
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--gpu-query-interval-sec", type=float, default=1.0)
    parser.add_argument("--baseline-seconds", type=float, default=10.0)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = get_mlebench_data_dir()
    if args.output_dir is None:
        args.output_dir = str(default_output_dir(args.task, Path(args.workload).stem, args.parallel_models, args.seed))

    if args.check_only:
        print(json.dumps(validate_prepared_task(args.data_dir, args.task), indent=2))
        return

    summary = execute_monitored_workload(args)
    print(json.dumps(summary, indent=2))
    print(f"Resource report: {Path(args.output_dir).resolve() / 'resource_monitor' / 'report.md'}")


if __name__ == "__main__":
    main()
