"""
Baseline CPU scheduling algorithms:
1. FCFS (First-Come, First-Served)
2. SJF / SRTF (Preemptive Shortest Remaining Time First)
3. Round Robin (Fixed Quantum)
4. RLBMCS / Dynamic Adaptive RR (Rinku et al. / Shrivastava 2010 - Published Baseline)
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .simulator import Simulator
from .process import Process, ProcessState, BurstType


class BaseScheduler:
    def __init__(self, name: str):
        self.name = name

    def run(self, simulator: Simulator) -> Dict[str, Any]:
        raise NotImplementedError


class FCFSScheduler(BaseScheduler):
    def __init__(self):
        super().__init__(name="FCFS")

    def run(self, sim: Simulator) -> Dict[str, Any]:
        while not sim.is_done() and sim.current_time < sim.max_simulation_ticks:
            sim.admit_new_arrivals()
            sim.step_io()

            # Handle context switch
            if sim.in_context_switch:
                sim.context_switch_remaining -= 1
                sim.context_switch_ticks += 1
                sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                sim.recent_cpu_ticks.append(0)
                if sim.context_switch_remaining <= 0:
                    sim.in_context_switch = False
                    sim.running_process = sim.pending_process
                    sim.pending_process = None
                    if sim.running_process:
                        sim.running_process.state = ProcessState.RUNNING
                        if sim.running_process.start_time is None:
                            sim.running_process.start_time = sim.current_time
                            sim.running_process.response_time = sim.current_time - sim.running_process.arrival_time
                sim.current_time += 1
                continue

            # Pick next process if idle
            if sim.running_process is None:
                if sim.ready_queue:
                    sim.ready_queue.sort(key=lambda p: (p.arrival_time, p.pid))
                    next_proc = sim.ready_queue.pop(0)
                    can_execute_now = sim.dispatch_process(next_proc)
                    if not can_execute_now:
                        sim.context_switch_ticks += 1
                        sim.context_switch_remaining -= 1
                        sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                        sim.recent_cpu_ticks.append(0)
                        if sim.context_switch_remaining <= 0:
                            sim.in_context_switch = False
                            sim.running_process = sim.pending_process
                            sim.pending_process = None
                            if sim.running_process:
                                sim.running_process.state = ProcessState.RUNNING
                                if sim.running_process.start_time is None:
                                    sim.running_process.start_time = sim.current_time + 1
                                    sim.running_process.response_time = (sim.current_time + 1) - sim.running_process.arrival_time
                        sim.current_time += 1
                        continue
                else:
                    sim.cpu_idle_ticks += 1
                    sim.recent_cpu_ticks.append(0)
                    sim._log_timeline(None, "IDLE", sim.current_time, sim.current_time + 1)
                    sim.current_time += 1
                    continue

            # Execute running process
            p = sim.running_process
            sim.cpu_busy_ticks += 1
            sim.recent_cpu_ticks.append(1)
            sim._log_timeline(p.pid, "CPU", sim.current_time, sim.current_time + 1)

            for ready_p in sim.ready_queue:
                ready_p.total_waiting_time += 1

            cpu_done = p.advance_cpu(1)

            if cpu_done:
                if p.is_terminated:
                    p.state = ProcessState.TERMINATED
                    p.finish_time = sim.current_time + 1
                    sim.completed_processes.append(p)
                    sim.recent_completions.append(1)
                elif p.current_burst and p.current_burst.burst_type == BurstType.IO:
                    p.state = ProcessState.BLOCKED_IO
                    sim.blocked_queue.append(p)
                else:
                    p.state = ProcessState.READY
                    p.last_ready_time = sim.current_time + 1
                    sim.ready_queue.append(p)
                sim.running_process = None

            sim.current_time += 1

        return {"name": self.name, "simulator": sim}


class SJFScheduler(BaseScheduler):
    """Preemptive Shortest Remaining Time First (SRTF)."""
    def __init__(self):
        super().__init__(name="SJF (SRTF)")

    def run(self, sim: Simulator) -> Dict[str, Any]:
        while not sim.is_done() and sim.current_time < sim.max_simulation_ticks:
            sim.admit_new_arrivals()
            sim.step_io()

            if sim.in_context_switch:
                sim.context_switch_remaining -= 1
                sim.context_switch_ticks += 1
                sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                sim.recent_cpu_ticks.append(0)
                if sim.context_switch_remaining <= 0:
                    sim.in_context_switch = False
                    sim.running_process = sim.pending_process
                    sim.pending_process = None
                    if sim.running_process:
                        sim.running_process.state = ProcessState.RUNNING
                        if sim.running_process.start_time is None:
                            sim.running_process.start_time = sim.current_time
                            sim.running_process.response_time = sim.current_time - sim.running_process.arrival_time
                sim.current_time += 1
                continue

            # Check preemption
            if sim.ready_queue:
                sim.ready_queue.sort(key=lambda p: (p.remaining_cpu_time, p.arrival_time, p.pid))
                shortest_ready = sim.ready_queue[0]

                if sim.running_process is not None:
                    if shortest_ready.remaining_cpu_time < sim.running_process.remaining_cpu_time:
                        curr = sim.running_process
                        curr.state = ProcessState.READY
                        curr.preemptions += 1
                        curr.last_ready_time = sim.current_time
                        sim.ready_queue.append(curr)
                        sim.running_process = None

                        sim.ready_queue.sort(key=lambda p: (p.remaining_cpu_time, p.arrival_time, p.pid))
                        next_proc = sim.ready_queue.pop(0)
                        can_execute = sim.dispatch_process(next_proc)
                        if not can_execute:
                            sim.context_switch_ticks += 1
                            sim.context_switch_remaining -= 1
                            sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                            sim.recent_cpu_ticks.append(0)
                            if sim.context_switch_remaining <= 0:
                                sim.in_context_switch = False
                                sim.running_process = sim.pending_process
                                sim.pending_process = None
                                if sim.running_process:
                                    sim.running_process.state = ProcessState.RUNNING
                                    if sim.running_process.start_time is None:
                                        sim.running_process.start_time = sim.current_time + 1
                                        sim.running_process.response_time = (sim.current_time + 1) - sim.running_process.arrival_time
                            sim.current_time += 1
                            continue

            if sim.running_process is None:
                if sim.ready_queue:
                    sim.ready_queue.sort(key=lambda p: (p.remaining_cpu_time, p.arrival_time, p.pid))
                    next_proc = sim.ready_queue.pop(0)
                    can_execute = sim.dispatch_process(next_proc)
                    if not can_execute:
                        sim.context_switch_ticks += 1
                        sim.context_switch_remaining -= 1
                        sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                        sim.recent_cpu_ticks.append(0)
                        if sim.context_switch_remaining <= 0:
                            sim.in_context_switch = False
                            sim.running_process = sim.pending_process
                            sim.pending_process = None
                            if sim.running_process:
                                sim.running_process.state = ProcessState.RUNNING
                                if sim.running_process.start_time is None:
                                    sim.running_process.start_time = sim.current_time + 1
                                    sim.running_process.response_time = (sim.current_time + 1) - sim.running_process.arrival_time
                        sim.current_time += 1
                        continue
                else:
                    sim.cpu_idle_ticks += 1
                    sim.recent_cpu_ticks.append(0)
                    sim._log_timeline(None, "IDLE", sim.current_time, sim.current_time + 1)
                    sim.current_time += 1
                    continue

            p = sim.running_process
            sim.cpu_busy_ticks += 1
            sim.recent_cpu_ticks.append(1)
            sim._log_timeline(p.pid, "CPU", sim.current_time, sim.current_time + 1)

            for ready_p in sim.ready_queue:
                ready_p.total_waiting_time += 1

            cpu_done = p.advance_cpu(1)

            if cpu_done:
                if p.is_terminated:
                    p.state = ProcessState.TERMINATED
                    p.finish_time = sim.current_time + 1
                    sim.completed_processes.append(p)
                    sim.recent_completions.append(1)
                elif p.current_burst and p.current_burst.burst_type == BurstType.IO:
                    p.state = ProcessState.BLOCKED_IO
                    sim.blocked_queue.append(p)
                else:
                    p.state = ProcessState.READY
                    p.last_ready_time = sim.current_time + 1
                    sim.ready_queue.append(p)
                sim.running_process = None

            sim.current_time += 1

        return {"name": self.name, "simulator": sim}


class RoundRobinScheduler(BaseScheduler):
    def __init__(self, time_quantum: int = 4):
        super().__init__(name=f"Round Robin (q={time_quantum})")
        self.time_quantum = time_quantum

    def run(self, sim: Simulator) -> Dict[str, Any]:
        quantum_spent = 0

        while not sim.is_done() and sim.current_time < sim.max_simulation_ticks:
            sim.admit_new_arrivals()
            sim.step_io()

            if sim.in_context_switch:
                sim.context_switch_remaining -= 1
                sim.context_switch_ticks += 1
                sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                sim.recent_cpu_ticks.append(0)
                if sim.context_switch_remaining <= 0:
                    sim.in_context_switch = False
                    sim.running_process = sim.pending_process
                    sim.pending_process = None
                    quantum_spent = 0
                    if sim.running_process:
                        sim.running_process.state = ProcessState.RUNNING
                        if sim.running_process.start_time is None:
                            sim.running_process.start_time = sim.current_time
                            sim.running_process.response_time = sim.current_time - sim.running_process.arrival_time
                sim.current_time += 1
                continue

            if sim.running_process is None:
                if sim.ready_queue:
                    next_proc = sim.ready_queue.pop(0)
                    can_execute = sim.dispatch_process(next_proc)
                    quantum_spent = 0
                    if not can_execute:
                        sim.context_switch_ticks += 1
                        sim.context_switch_remaining -= 1
                        sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                        sim.recent_cpu_ticks.append(0)
                        if sim.context_switch_remaining <= 0:
                            sim.in_context_switch = False
                            sim.running_process = sim.pending_process
                            sim.pending_process = None
                            if sim.running_process:
                                sim.running_process.state = ProcessState.RUNNING
                                if sim.running_process.start_time is None:
                                    sim.running_process.start_time = sim.current_time + 1
                                    sim.running_process.response_time = (sim.current_time + 1) - sim.running_process.arrival_time
                        sim.current_time += 1
                        continue
                else:
                    sim.cpu_idle_ticks += 1
                    sim.recent_cpu_ticks.append(0)
                    sim._log_timeline(None, "IDLE", sim.current_time, sim.current_time + 1)
                    sim.current_time += 1
                    continue

            p = sim.running_process
            sim.cpu_busy_ticks += 1
            sim.recent_cpu_ticks.append(1)
            sim._log_timeline(p.pid, "CPU", sim.current_time, sim.current_time + 1)

            for ready_p in sim.ready_queue:
                ready_p.total_waiting_time += 1

            cpu_done = p.advance_cpu(1)
            quantum_spent += 1

            if cpu_done:
                if p.is_terminated:
                    p.state = ProcessState.TERMINATED
                    p.finish_time = sim.current_time + 1
                    sim.completed_processes.append(p)
                    sim.recent_completions.append(1)
                elif p.current_burst and p.current_burst.burst_type == BurstType.IO:
                    p.state = ProcessState.BLOCKED_IO
                    sim.blocked_queue.append(p)
                else:
                    p.state = ProcessState.READY
                    p.last_ready_time = sim.current_time + 1
                    sim.ready_queue.append(p)
                sim.running_process = None
                quantum_spent = 0
            elif quantum_spent >= self.time_quantum:
                p.state = ProcessState.READY
                p.preemptions += 1
                p.last_ready_time = sim.current_time + 1
                sim.ready_queue.append(p)
                sim.running_process = None
                quantum_spent = 0

            sim.current_time += 1

        return {"name": self.name, "simulator": sim}


class RLBMCS_AdaptiveRRScheduler(BaseScheduler):
    """
    Dynamic Adaptive Round Robin (RLBMCS / Rinku et al. & Shrivastava 2010):
    Published literature baseline: adapts time quantum based on ready-queue burst characteristics,
    e.g., dynamic quantum = mean burst time in ready queue.
    Fixed aging alpha=0 and no multi-objective fairness feedback.
    """
    def __init__(self, min_quantum: int = 2, max_quantum: int = 16):
        super().__init__(name="RLBMCS / Dyn-RR (Lit. Baseline)")
        self.min_quantum = min_quantum
        self.max_quantum = max_quantum
        self.current_quantum = 4

    def update_quantum(self, ready_queue: List[Process]):
        if ready_queue:
            bursts = [p.current_burst.remaining for p in ready_queue if p.current_burst]
            if bursts:
                mean_burst = float(np.mean(bursts))
                adapted_q = int(round(mean_burst * 0.5))
                self.current_quantum = max(self.min_quantum, min(self.max_quantum, adapted_q))

    def run(self, sim: Simulator) -> Dict[str, Any]:
        quantum_spent = 0

        while not sim.is_done() and sim.current_time < sim.max_simulation_ticks:
            sim.admit_new_arrivals()
            sim.step_io()

            if sim.in_context_switch:
                sim.context_switch_remaining -= 1
                sim.context_switch_ticks += 1
                sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                sim.recent_cpu_ticks.append(0)
                if sim.context_switch_remaining <= 0:
                    sim.in_context_switch = False
                    sim.running_process = sim.pending_process
                    sim.pending_process = None
                    quantum_spent = 0
                    if sim.running_process:
                        sim.running_process.state = ProcessState.RUNNING
                        if sim.running_process.start_time is None:
                            sim.running_process.start_time = sim.current_time
                            sim.running_process.response_time = sim.current_time - sim.running_process.arrival_time
                sim.current_time += 1
                continue

            if sim.running_process is None:
                if sim.ready_queue:
                    self.update_quantum(sim.ready_queue)
                    next_proc = sim.ready_queue.pop(0)
                    can_execute = sim.dispatch_process(next_proc)
                    quantum_spent = 0
                    if not can_execute:
                        sim.context_switch_ticks += 1
                        sim.context_switch_remaining -= 1
                        sim._log_timeline(None, "CS", sim.current_time, sim.current_time + 1)
                        sim.recent_cpu_ticks.append(0)
                        if sim.context_switch_remaining <= 0:
                            sim.in_context_switch = False
                            sim.running_process = sim.pending_process
                            sim.pending_process = None
                            if sim.running_process:
                                sim.running_process.state = ProcessState.RUNNING
                                if sim.running_process.start_time is None:
                                    sim.running_process.start_time = sim.current_time + 1
                                    sim.running_process.response_time = (sim.current_time + 1) - sim.running_process.arrival_time
                        sim.current_time += 1
                        continue
                else:
                    sim.cpu_idle_ticks += 1
                    sim.recent_cpu_ticks.append(0)
                    sim._log_timeline(None, "IDLE", sim.current_time, sim.current_time + 1)
                    sim.current_time += 1
                    continue

            p = sim.running_process
            sim.cpu_busy_ticks += 1
            sim.recent_cpu_ticks.append(1)
            sim._log_timeline(p.pid, "CPU", sim.current_time, sim.current_time + 1)

            for ready_p in sim.ready_queue:
                ready_p.total_waiting_time += 1

            cpu_done = p.advance_cpu(1)
            quantum_spent += 1

            if cpu_done:
                if p.is_terminated:
                    p.state = ProcessState.TERMINATED
                    p.finish_time = sim.current_time + 1
                    sim.completed_processes.append(p)
                    sim.recent_completions.append(1)
                elif p.current_burst and p.current_burst.burst_type == BurstType.IO:
                    p.state = ProcessState.BLOCKED_IO
                    sim.blocked_queue.append(p)
                else:
                    p.state = ProcessState.READY
                    p.last_ready_time = sim.current_time + 1
                    sim.ready_queue.append(p)
                sim.running_process = None
                quantum_spent = 0
            elif quantum_spent >= self.current_quantum:
                p.state = ProcessState.READY
                p.preemptions += 1
                p.last_ready_time = sim.current_time + 1
                sim.ready_queue.append(p)
                sim.running_process = None
                quantum_spent = 0

            sim.current_time += 1

        return {"name": self.name, "simulator": sim}
