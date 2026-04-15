# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

_ACTIVE_MONITOR: Any | None = None


def set_active_monitor(monitor: Any | None) -> None:
    global _ACTIVE_MONITOR
    _ACTIVE_MONITOR = monitor


def get_active_monitor() -> Any | None:
    return _ACTIVE_MONITOR


@contextmanager
def resource_phase(phase: str, **metadata: Any) -> Iterator[None]:
    monitor = get_active_monitor()
    if monitor is None:
        with nullcontext():
            yield
        return

    with monitor.phase(phase, **metadata):
        yield


def register_process(pid: int | None, label: str | None = None) -> None:
    if pid is None:
        return
    monitor = get_active_monitor()
    if monitor is not None:
        monitor.register_process(pid, label=label)


def unregister_process(pid: int | None) -> None:
    if pid is None:
        return
    monitor = get_active_monitor()
    if monitor is not None:
        monitor.unregister_process(pid)
