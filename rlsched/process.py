"""
Process and Burst definitions for the RL-Sched discrete-event simulator.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ProcessState(Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED_IO = "BLOCKED_IO"
    TERMINATED = "TERMINATED"


class BurstType(Enum):
    CPU = "CPU"
    IO = "IO"


@dataclass
class Burst:
    burst_type: BurstType
    duration: int
    remaining: int

    def __init__(self, burst_type: BurstType, duration: int):
        self.burst_type = burst_type
        self.duration = duration
        self.remaining = duration


@dataclass
class Process:
    pid: int
    arrival_time: int
    bursts: List[Burst]
    base_priority: float = 0.0
    
    # Dynamic state tracking
    current_burst_index: int = 0
    state: ProcessState = ProcessState.READY
    dynamic_priority: float = 0.0
    
    # Timing and accounting metrics
    start_time: Optional[int] = None
    finish_time: Optional[int] = None
    total_waiting_time: int = 0
    total_cpu_time: int = 0
    total_io_time: int = 0
    last_ready_time: int = 0
    response_time: Optional[int] = None
    preemptions: int = 0

    def __post_init__(self):
        self.dynamic_priority = self.base_priority
        self.last_ready_time = self.arrival_time

    @property
    def is_terminated(self) -> bool:
        return self.current_burst_index >= len(self.bursts)

    @property
    def current_burst(self) -> Optional[Burst]:
        if self.is_terminated:
            return None
        return self.bursts[self.current_burst_index]

    @property
    def remaining_cpu_time(self) -> int:
        """Sum of all remaining CPU bursts for this process."""
        total = 0
        for i in range(self.current_burst_index, len(self.bursts)):
            b = self.bursts[i]
            if b.burst_type == BurstType.CPU:
                total += b.remaining
        return total

    @property
    def total_initial_cpu_time(self) -> int:
        return sum(b.duration for b in self.bursts if b.burst_type == BurstType.CPU)

    @property
    def total_initial_io_time(self) -> int:
        return sum(b.duration for b in self.bursts if b.burst_type == BurstType.IO)

    @property
    def turnaround_time(self) -> Optional[int]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    def advance_cpu(self, ticks: int = 1) -> bool:
        """
        Advances CPU execution by `ticks`.
        Returns True if the current CPU burst finished, False otherwise.
        """
        curr = self.current_burst
        if curr is None or curr.burst_type != BurstType.CPU:
            raise ValueError(f"Process {self.pid} is not on a CPU burst.")

        actual_exec = min(ticks, curr.remaining)
        curr.remaining -= actual_exec
        self.total_cpu_time += actual_exec

        if curr.remaining == 0:
            self.current_burst_index += 1
            return True
        return False

    def advance_io(self, ticks: int = 1) -> bool:
        """
        Advances I/O execution by `ticks`.
        Returns True if the current I/O burst finished, False otherwise.
        """
        curr = self.current_burst
        if curr is None or curr.burst_type != BurstType.IO:
            raise ValueError(f"Process {self.pid} is not on an I/O burst.")

        actual_io = min(ticks, curr.remaining)
        curr.remaining -= actual_io
        self.total_io_time += actual_io

        if curr.remaining == 0:
            self.current_burst_index += 1
            return True
        return False

    def update_dynamic_priority(self, aging_rate: float, current_time: int):
        """
        Calculates priority using aging formula:
        dynamic_priority = base_priority + aging_rate * time_spent_in_ready_queue
        """
        if self.state == ProcessState.READY:
            wait_duration = current_time - self.last_ready_time
            self.dynamic_priority = self.base_priority + aging_rate * wait_duration
        else:
            self.dynamic_priority = self.base_priority

    def clone(self) -> 'Process':
        """Deep clone process for repeatable apples-to-apples simulation comparisons."""
        cloned_bursts = [Burst(b.burst_type, b.duration) for b in self.bursts]
        p = Process(
            pid=self.pid,
            arrival_time=self.arrival_time,
            bursts=cloned_bursts,
            base_priority=self.base_priority
        )
        return p
