"""Optional component timing support for training models."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Iterator

import torch


class TrainingTimingMixin:
    """Add non-blocking training component timing to a model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._training_timing_enabled = False
        self._current_cuda_training_timings: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._current_cpu_training_timings: dict[str, float] = defaultdict(float)
        self._pending_training_timing_groups: deque[
            tuple[
                list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
                dict[str, float],
            ]
        ] = deque()

    def set_training_timing_enabled(self, enabled: bool) -> None:
        """Enable lightweight component timing during training."""
        self._training_timing_enabled = bool(enabled)

    @contextmanager
    def training_timer(self, name: str, *, device: torch.device) -> Iterator[None]:
        """Record one component without synchronizing the CUDA hot path."""
        if not self._training_timing_enabled or not self.training:
            yield
            return

        if device.type == "cuda" and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._current_cuda_training_timings.append((name, start, end))
            return

        started = time.perf_counter()
        try:
            yield
        finally:
            self._current_cpu_training_timings[name] += time.perf_counter() - started

    def finalize_training_timing_step(self) -> None:
        """Close the current optimizer-step timing group without waiting."""
        if not self._training_timing_enabled:
            return
        if (
            not self._current_cuda_training_timings
            and not self._current_cpu_training_timings
        ):
            return
        self._pending_training_timing_groups.append(
            (
                self._current_cuda_training_timings,
                dict(self._current_cpu_training_timings),
            )
        )
        self._current_cuda_training_timings = []
        self._current_cpu_training_timings.clear()

    def pop_completed_training_timings(self) -> list[dict[str, float]]:
        """Return completed optimizer-step timing groups without blocking."""
        completed: list[dict[str, float]] = []
        while self._pending_training_timing_groups:
            cuda_timings, cpu_timings = self._pending_training_timing_groups[0]
            if any(not end.query() for _, _, end in cuda_timings):
                break
            self._pending_training_timing_groups.popleft()
            values: dict[str, float] = defaultdict(float)
            values.update(cpu_timings)
            for name, start, end in cuda_timings:
                values[name] += float(start.elapsed_time(end)) / 1000.0
            completed.append(dict(values))
        return completed
