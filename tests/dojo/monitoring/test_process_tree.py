import os
import subprocess
import sys

from dojo.monitoring.process_tree import aggregate_process_tree


def test_process_tree_aggregation_includes_registered_child() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        sample = aggregate_process_tree(root_pid=os.getpid(), extra_pids=[child.pid], ts=123.0)
    finally:
        child.terminate()
        child.wait(timeout=10)

    assert child.pid in sample["pids"]
    assert sample["root_pid"] == os.getpid()
    assert sample["rss_mb_total"] > 0
