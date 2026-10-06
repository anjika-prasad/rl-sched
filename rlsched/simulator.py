"""
Discrete-Event OS Simulator Engine for RL-Sched.
Simulates CPU execution, ready queues, parallel I/O blocking, context switches,
and provides telemetry hooks for adaptive scheduling policies.
"""

from typing import List, Optional, Dict, Any
from collections import deque
import numpy as np

from .process import Process, ProcessState, BurstType


class Simulator:
    def __init__(
        self,
        workload: List[Process],
        context_switch_cost: int = 1,
        max_simulation_ticks: int = 100000
    ):
        # Workload cloned so repeated runs are clean
        self.unstarted_processes: List[Process] = sorted(
            [p.clone() for p in workload],
            key=lambda x: (x.arrival_time, x.pid)
        )
        self.total_processes = len(self.unstarted_processes)
        self.context_switch_cost = context_switch_cost
        self.max_simulation_ticks = max_simulation_ticks

        # Queues & State
        self.current_time: int = 0
        self.ready_queue: List[Process] = []
        self.blocked_queue: List[Process] = []
        self.running_process: Optional[Process] = None
        self.completed_processes: List[Process] = []

        # Context switch state
        self.in_context_switch: bool = False
        self.context_switch_remaining: int = 0
        self.pending_process: Optional[Process] = None

        # Accounting & Timeline
        self.cpu_busy_ticks: int = 0
        self.context_switch_ticks: int = 0
        self.cpu_idle_ticks: int = 0
        self.timeline: List[Dict[str, Any]] = []

        # Rolling telemetry tracking for RL agent
        self.history_window = 30
        self.recent_cpu_ticks = deque(maxlen=self.history_window)
        self.recent_completions = deque(maxlen=self.history_window)

    def get_system_telemetry(self) -> Dict[str, float]:
        """Calculates instantaneous and windowed state metrics for the scheduler."""
        if len(self.recent_cpu_ticks) > 0:
            cpu_util = sum(self.recent_cpu_ticks) / len(self.recent_cpu_ticks)
        else:
            cpu_util = 0.0

        q_len = len(self.ready_queue)

        if q_len > 0:
            avg_burst = sum(p.current_burst.remaining for p in self.ready_queue if p.current_burst) / q_len
        elif self.running_process and self.running_process.current_burst:
            avg_burst = float(self.running_process.current_burst.remaining)
        else:
            avg_burst = 0.0

        active = self.ready_queue + self.blocked_queue + ([self.running_process] if self.running_process else [])
        if active:
            blocked_count = len(self.blocked_queue)
            io_ratio = blocked_count / len(active)
        else:
            io_ratio = 0.0

        wait_times = [self.current_time - p.last_ready_time for p in self.ready_queue]
        if wait_times:
            avg_wait = float(np.mean(wait_times))
            wait_var = float(np.var(wait_times))
            max_wait = float(np.max(wait_times))
        else:
            avg_wait = 0.0
            wait_var = 0.0
            max_wait = 0.0

        throughput = sum(self.recent_completions) / max(1, len(self.recent_completions))

        return {
            "current_time": self.current_time,
            "cpu_utilization": cpu_util,
            "ready_queue_length": q_len,
            "avg_burst_estimate": avg_burst,
            "io_ratio": io_ratio,
            "avg_current_wait": avg_wait,
            "wait_variance": wait_var,
            "max_current_wait": max_wait,
            "throughput": throughput,
            "completed_count": len(self.completed_processes),
            "blocked_count": len(self.blocked_queue)
        }

    def _log_timeline(self, pid: Optional[int], event_type: str, start: int, end: int):
        if end > start:
            if self.timeline and self.timeline[-1]["pid"] == pid and self.timeline[-1]["type"] == event_type and self.timeline[-1]["end"] == start:
                self.timeline[-1]["end"] = end
            else:
                self.timeline.append({
                    "start": start,
                    "end": end,
                    "pid": pid,
                    "type": event_type
                })

    def admit_new_arrivals(self):
        """Moves processes whose arrival_time <= current_time into ready_queue."""
        arrivals: List[Process] = []
        while self.unstarted_processes and self.unstarted_processes[0].arrival_time <= self.current_time:
            p = self.unstarted_processes.pop(0)
            p.state = ProcessState.READY
            p.last_ready_time = self.current_time
            self.ready_queue.append(p)
            arrivals.append(p)
        return arrivals

    def step_io(self):
        """Advances I/O for all blocked processes in parallel."""
        still_blocked: List[Process] = []
        for p in self.blocked_queue:
            if p.current_burst and p.current_burst.burst_type == BurstType.IO:
                io_finished = p.advance_io(1)
                if io_finished:
                    if p.is_terminated:
                        p.state = ProcessState.TERMINATED
                        p.finish_time = self.current_time
                        self.completed_processes.append(p)
                        self.recent_completions.append(1)
                    elif p.current_burst.burst_type == BurstType.CPU:
                        p.state = ProcessState.READY
                        p.last_ready_time = self.current_time
                        self.ready_queue.append(p)
                    else:
                        still_blocked.append(p)
                else:
                    still_blocked.append(p)
            else:
                # In case a CPU burst process was in blocked queue, re-route
                if p.current_burst and p.current_burst.burst_type == BurstType.CPU:
                    p.state = ProcessState.READY
                    p.last_ready_time = self.current_time
                    self.ready_queue.append(p)
                elif p.is_terminated:
                    p.state = ProcessState.TERMINATED
                    p.finish_time = self.current_time
                    self.completed_processes.append(p)
                    self.recent_completions.append(1)

        self.blocked_queue = still_blocked

    def dispatch_process(self, target_process: Process) -> bool:
        """
        Dispatches target_process to CPU.
        If context switch cost > 0, enters context switch state and returns False (tick consumed by CS).
        If context switch cost == 0, directly makes process RUNNING and returns True (can execute in same tick).
        """
        if self.context_switch_cost > 0:
            self.in_context_switch = True
            self.context_switch_remaining = self.context_switch_cost
            self.pending_process = target_process
            return False
        else:
            self.in_context_switch = False
            self.running_process = target_process
            self.running_process.state = ProcessState.RUNNING
            if self.running_process.start_time is None:
                self.running_process.start_time = self.current_time
                self.running_process.response_time = self.current_time - self.running_process.arrival_time
            return True

    def is_done(self) -> bool:
        """Returns True if all processes are terminated."""
        return (
            len(self.completed_processes) == self.total_processes and
            not self.running_process and
            not self.in_context_switch and
            len(self.unstarted_processes) == 0 and
            len(self.ready_queue) == 0 and
            len(self.blocked_queue) == 0
        )
