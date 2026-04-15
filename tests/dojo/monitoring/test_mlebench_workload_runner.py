import argparse
import ast
import os
from pathlib import Path

from dojo.monitoring import mlebench_workload_runner as runner


def test_builtin_workloads_are_valid_python() -> None:
    for path in runner.BUILTIN_WORKLOADS.values():
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        assert "submission.csv" in source


def test_validate_prepared_task_accepts_minimal_aerial_cactus_layout(tmp_path) -> None:
    public_dir = tmp_path / runner.TASK_NAME / "prepared" / "public"
    private_dir = tmp_path / runner.TASK_NAME / "prepared" / "private"
    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    (public_dir / "train.csv").write_text("id,has_cactus\nx.jpg,1\n", encoding="utf-8")
    (public_dir / "sample_submission.csv").write_text("id,has_cactus\ny.jpg,0\n", encoding="utf-8")

    result = runner.validate_prepared_task(tmp_path)

    assert result["public_dir"] == str(public_dir)
    assert result["private_dir"] == str(private_dir)
    assert result["train_csv"].endswith("train.csv")
    assert result["sample_submission"].endswith("sample_submission.csv")


def test_validate_prepared_task_accepts_histopathologic_train_labels_layout(tmp_path) -> None:
    task_name = runner.HISTOPATHOLOGIC_TASK_NAME
    public_dir = tmp_path / task_name / "prepared" / "public"
    private_dir = tmp_path / task_name / "prepared" / "private"
    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    (public_dir / "train_labels.csv").write_text("id,label\nx,1\n", encoding="utf-8")
    (public_dir / "sample_submission.csv").write_text("id,label\ny,0\n", encoding="utf-8")

    result = runner.validate_prepared_task(tmp_path, task_name)

    assert result["train_file"].endswith("train_labels.csv")
    assert result["train_csv"].endswith("train_labels.csv")
    assert result["sample_submission"].endswith("sample_submission.csv")


def test_workload_env_and_temporary_env_restore() -> None:
    args = argparse.Namespace(
        parallel_models=4,
        epochs=8,
        batch_size=256,
        image_size=64,
        dataloader_workers=2,
        seed=123,
        train_repeat=3,
    )
    env = runner.build_workload_env(args)

    assert env["AERIAL_CACTUS_PARALLEL_MODELS"] == "4"
    assert env["AERIAL_CACTUS_EPOCHS"] == "8"
    assert env["AERIAL_CACTUS_TRAIN_REPEAT"] == "3"
    assert env["HISTOPATHOLOGIC_PARALLEL_MODELS"] == "4"
    assert env["MLEBENCH_WORKLOAD_PARALLEL_MODELS"] == "4"

    os.environ["AERIAL_CACTUS_PARALLEL_MODELS"] = "old"
    with runner.temporary_env(env):
        assert os.environ["AERIAL_CACTUS_PARALLEL_MODELS"] == "4"
    assert os.environ["AERIAL_CACTUS_PARALLEL_MODELS"] == "old"
    os.environ.pop("AERIAL_CACTUS_PARALLEL_MODELS")


def test_workload_path_resolves_builtin_and_custom(tmp_path) -> None:
    custom = tmp_path / "custom.py"
    custom.write_text("print('ok')\n", encoding="utf-8")

    assert runner.workload_path("parallel") == runner.BUILTIN_WORKLOADS["parallel"]
    assert (
        runner.workload_path("parallel", runner.HISTOPATHOLOGIC_TASK_NAME)
        == runner.BUILTIN_WORKLOADS["histopathologic_parallel"]
    )
    assert runner.workload_path("histopathologic_parallel") == runner.BUILTIN_WORKLOADS["histopathologic_parallel"]
    assert runner.workload_path(str(custom)) == custom.resolve()


def test_default_output_dir_names_task_workload_parallelism_and_seed() -> None:
    output = runner.default_output_dir(runner.TASK_NAME, "parallel", 4, 123)

    assert isinstance(output, Path)
    assert runner.TASK_NAME in str(output)
    assert "parallel" in str(output)
    assert "p4" in str(output)
    assert "seed123" in str(output)
