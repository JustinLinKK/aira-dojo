# Resource Monitoring

Dojo can write a local resource-monitoring report for MLEBench runs. The monitor samples the Dojo process tree, registered interpreter child processes, GPU totals from `nvidia-smi`, and GPU compute-app process rows. It also records framework phase events so the report can distinguish generated solution training from setup, LLM/operator work, scoring, and cleanup.

Monitoring artifacts are written under:

```text
{logger.output_dir}/resource_monitor/
```

The directory contains raw JSONL samples, `metadata.json`, `summary.json`, `report.md`, and plots for CPU, RSS, GPU utilization, GPU memory, and GPU power.

## Example

```bash
python -m dojo.main_run \
  +_exp=run_example \
  interpreter=python \
  logger.use_wandb=False \
  solver.step_limit=1 \
  monitoring.enabled=true \
  monitoring.sample_interval_sec=1
```

To regenerate a report from existing raw files:

```bash
python -m dojo.monitoring.report /path/to/run/resource_monitor
```

## Interpreting The Report

The primary training window is the `generated_solution_execute` phase, which wraps execution of generated MLEBench solution code. GPU memory findings are reported as idle reserved GPU memory because allocated VRAM is not inherently wasteful; it becomes suspicious when VRAM remains high while GPU compute utilization is near zero for sustained periods.

For stronger evidence, compare:

- one no-op or baseline run,
- one simple MLEBench task with `solver.step_limit=1`,
- three repeated seeds of the same task,
- one known GPU-heavy generated solution if available.

The baseline phase is collected before `framework_total`, so display/driver VRAM, idle GPU watts, and background CPU usage are not counted as framework runtime.

## Deterministic MLE-Bench Workloads

For deterministic GPU profiling without LLM variation, use the built-in MLE-Bench workload runner. It executes fixed generated-solution code through `MLEBenchTask.step_task`, so the report still captures the primary `generated_solution_execute` phase.

For a long GPU training run on a 32 GB RTX 5090 class machine, prefer `histopathologic-cancer-detection`. It has about 174k train patches and is large enough to make `generated_solution_execute` dominate the report. `aerial-cactus-identification` is only useful as a smoke test.

Prepare the histopathology task:

```bash
PYTHONPATH=src python src/dojo/tasks/mlebench/utils/prepare.py \
  -c histopathologic-cancer-detection \
  --data-dir=/workspaces/aira-dojo/.cache/mlebench

export MLE_BENCH_DATA_DIR=/workspaces/aira-dojo/.cache/mlebench
```

Check that the prepared task is visible:

```bash
PYTHONPATH=src python -m dojo.monitoring.mlebench_workload_runner \
  --check-only \
  --task histopathologic-cancer-detection \
  --data-dir=/workspaces/aira-dojo/.cache/mlebench
```

Run the long four-model histopathology workload:

```bash
PYTHONPATH=src python -m dojo.monitoring.mlebench_workload_runner \
  --task histopathologic-cancer-detection \
  --workload parallel \
  --data-dir=/workspaces/aira-dojo/.cache/mlebench \
  --output-dir=/workspaces/aira-dojo/outputs/mlebench_workloads/histopathologic_p4_efficiency \
  --parallel-models 4 \
  --epochs 3 \
  --batch-size 512 \
  --image-size 96 \
  --dataloader-workers 0 \
  --train-repeat 1 \
  --baseline-seconds 10 \
  --sample-interval-sec 1 \
  --gpu-query-interval-sec 1 \
  --timeout-hours 8
```

For this devcontainer, `/dev/shm` is often very small, so use `--dataloader-workers 0`. The histopathology workload also auto-disables multiprocessing DataLoader workers when `/dev/shm` is below 1 GiB. For a stronger stress run, increase only one knob at a time: `--parallel-models 6`, then `--parallel-models 8`, then `--train-repeat 2`.

Prepare the smaller aerial cactus smoke task:

```bash
python src/dojo/tasks/mlebench/utils/prepare.py \
  -c aerial-cactus-identification \
  --data-dir=/path/to/mlebench

export MLE_BENCH_DATA_DIR=/path/to/mlebench
```

Check that the prepared task is visible:

```bash
PYTHONPATH=src python -m dojo.monitoring.mlebench_workload_runner \
  --check-only \
  --task aerial-cactus-identification
```

Run the monitor smoke test with a trivial valid submission:

```bash
PYTHONPATH=src python -m dojo.monitoring.mlebench_workload_runner \
  --workload noop \
  --baseline-seconds 3 \
  --sample-interval-sec 1
```

Run the single-model calibration:

```bash
PYTHONPATH=src python -m dojo.monitoring.mlebench_workload_runner \
  --workload parallel \
  --parallel-models 1 \
  --epochs 8 \
  --batch-size 256 \
  --baseline-seconds 10
```

Run the acceptance workload with four concurrent models:

```bash
PYTHONPATH=src python -m dojo.monitoring.mlebench_workload_runner \
  --workload parallel \
  --parallel-models 4 \
  --epochs 8 \
  --batch-size 256 \
  --dataloader-workers 2 \
  --baseline-seconds 10
```

Run the stress workload:

```bash
PYTHONPATH=src python -m dojo.monitoring.mlebench_workload_runner \
  --workload parallel \
  --parallel-models 8 \
  --epochs 8 \
  --batch-size 256 \
  --dataloader-workers 2 \
  --baseline-seconds 10
```

For repeatability, run the four-model command with three seeds:

```bash
for seed in 101 202 303; do
  PYTHONPATH=src python -m dojo.monitoring.mlebench_workload_runner \
    --workload parallel \
    --parallel-models 4 \
    --epochs 8 \
    --batch-size 256 \
    --seed "$seed" \
    --baseline-seconds 10
done
```

Each run writes:

```text
outputs/mlebench_workloads/<run-id>/
  workload_run_config.json
  workload_result.json
  workspace_agent/
  results/
  resource_monitor/report.md
  resource_monitor/summary.json
```

If the four-model run underutilizes the GPU, increase `--image-size`, `--epochs`, or `--train-repeat` before switching tasks. The intended fallback task is `plant-pathology-2020-fgvc7`.
