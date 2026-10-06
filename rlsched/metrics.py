"""
Metrics calculation module for RL-Sched.
Computes performance, responsiveness, throughput, and Jain's fairness index.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .process import Process


def calculate_jains_fairness_index(values: List[float]) -> float:
    """
    Computes Jain's Fairness Index:
    J(x) = (sum(x_i))^2 / (n * sum(x_i^2))
    Range is [1/n, 1.0], where 1.0 represents perfect equality.
    """
    if not values:
        return 1.0
    arr = np.array(values, dtype=float)
    # Avoid zero division if all values are 0
    if np.all(arr == 0):
        return 1.0
    # To evaluate fairness on waiting times, lower wait is better;
    # We can measure fairness of normalized turnaround efficiency or directly on wait times.
    # For waiting times, direct Jain's index on (wait + 1) measures parity in delay:
    sum_val = np.sum(arr)
    sum_sq = np.sum(arr ** 2)
    n = len(arr)
    if sum_sq == 0:
        return 1.0
    return float((sum_val ** 2) / (n * sum_sq))


def calculate_turnaround_ratio_fairness(processes: List[Process]) -> float:
    """
    Fairness based on Normalized Turnaround Time (TAT / Total CPU time).
    Ideally, all processes should experience a similar expansion factor regardless of job length.
    """
    ratios = []
    for p in processes:
        if p.finish_time is not None:
            cpu_time = max(1, p.total_initial_cpu_time)
            ratios.append(p.turnaround_time / cpu_time)
    return calculate_jains_fairness_index(ratios)


def evaluate_simulation(sim) -> Dict[str, Any]:
    """
    Evaluates a completed simulation run and returns summary metrics.
    """
    completed = sim.completed_processes
    total_completed = len(completed)
    total_ticks = max(1, sim.current_time)

    if total_completed == 0:
        return {
            "completed_count": 0,
            "total_time": total_ticks,
            "avg_waiting_time": 0.0,
            "avg_turnaround_time": 0.0,
            "avg_response_time": 0.0,
            "cpu_utilization": 0.0,
            "throughput": 0.0,
            "jains_fairness": 0.0,
            "starvation_count": 0,
            "context_switches": 0
        }

    wait_times = [p.total_waiting_time for p in completed]
    tat_times = [p.turnaround_time for p in completed if p.turnaround_time is not None]
    resp_times = [p.response_time for p in completed if p.response_time is not None]
    preemptions = sum(p.preemptions for p in completed)

    avg_wait = float(np.mean(wait_times))
    std_wait = float(np.std(wait_times))
    avg_tat = float(np.mean(tat_times))
    avg_resp = float(np.mean(resp_times)) if resp_times else 0.0

    # CPU utilization percentage
    cpu_util = (sim.cpu_busy_ticks / total_ticks) * 100.0

    # System throughput: completed processes per 100 ticks
    throughput = (total_completed / total_ticks) * 100.0

    # Jain's fairness on normalized turnaround ratios (standard OS fairness metric)
    jains_fairness = calculate_turnaround_ratio_fairness(completed)

    # Starvation count: processes waiting > (avg_wait + 2 * std_wait)
    starve_thresh = avg_wait + 2.0 * std_wait
    starvation_count = sum(1 for w in wait_times if w > starve_thresh and w > 20)

    # Context switches
    cs_ticks = sim.context_switch_ticks

    return {
        "completed_count": total_completed,
        "total_time": total_ticks,
        "avg_waiting_time": round(avg_wait, 2),
        "std_waiting_time": round(std_wait, 2),
        "avg_turnaround_time": round(avg_tat, 2),
        "avg_response_time": round(avg_resp, 2),
        "cpu_utilization": round(cpu_util, 2),
        "throughput": round(throughput, 4),
        "jains_fairness": round(jains_fairness, 4),
        "starvation_count": starvation_count,
        "preemptions": preemptions,
        "context_switch_ticks": cs_ticks
    }
