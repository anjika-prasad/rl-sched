"""
Workload generation module for RL-Sched.
Provides both static traces and dynamic multi-phase traces with shifting workload characteristics.
"""

import random
from typing import List
from .process import Process, Burst, BurstType


def generate_static_cpu_bound(
    num_processes: int = 30,
    arrival_rate: float = 0.5,
    min_cpu_burst: int = 15,
    max_cpu_burst: int = 40,
    seed: int = 42
) -> List[Process]:
    """
    Generates a CPU-bound workload with long compute bursts and minimal I/O.
    """
    rng = random.Random(seed)
    processes: List[Process] = []
    current_arrival = 0

    for pid in range(num_processes):
        inter_arrival = int(rng.expovariate(arrival_rate)) if arrival_rate > 0 else 0
        current_arrival += max(0, inter_arrival)

        num_cycles = rng.randint(1, 3)
        bursts: List[Burst] = []
        for c in range(num_cycles):
            cpu_duration = rng.randint(min_cpu_burst, max_cpu_burst)
            bursts.append(Burst(BurstType.CPU, cpu_duration))
            if c < num_cycles - 1 and rng.random() < 0.2:
                io_duration = rng.randint(2, 6)
                bursts.append(Burst(BurstType.IO, io_duration))

        processes.append(Process(pid=pid, arrival_time=current_arrival, bursts=bursts))

    return processes


def generate_static_io_bound(
    num_processes: int = 30,
    arrival_rate: float = 0.8,
    min_cpu_burst: int = 2,
    max_cpu_burst: int = 6,
    min_io_burst: int = 8,
    max_io_burst: int = 20,
    seed: int = 42
) -> List[Process]:
    """
    Generates an I/O-bound interactive workload with short compute bursts and frequent I/O waits.
    """
    rng = random.Random(seed)
    processes: List[Process] = []
    current_arrival = 0

    for pid in range(num_processes):
        inter_arrival = int(rng.expovariate(arrival_rate)) if arrival_rate > 0 else 0
        current_arrival += max(0, inter_arrival)

        num_cycles = rng.randint(3, 7)
        bursts: List[Burst] = []
        for c in range(num_cycles):
            cpu_duration = rng.randint(min_cpu_burst, max_cpu_burst)
            bursts.append(Burst(BurstType.CPU, cpu_duration))
            if c < num_cycles - 1:
                io_duration = rng.randint(min_io_burst, max_io_burst)
                bursts.append(Burst(BurstType.IO, io_duration))

        processes.append(Process(pid=pid, arrival_time=current_arrival, bursts=bursts))

    return processes


def generate_static_mixed(
    num_processes: int = 40,
    cpu_bound_ratio: float = 0.5,
    arrival_rate: float = 0.6,
    seed: int = 42
) -> List[Process]:
    """
    Generates a heterogeneous mixed workload containing both CPU-bound and I/O-bound processes.
    """
    rng = random.Random(seed)
    processes: List[Process] = []
    current_arrival = 0

    for pid in range(num_processes):
        inter_arrival = int(rng.expovariate(arrival_rate)) if arrival_rate > 0 else 0
        current_arrival += max(0, inter_arrival)

        is_cpu_bound = rng.random() < cpu_bound_ratio
        bursts: List[Burst] = []
        if is_cpu_bound:
            num_cycles = rng.randint(1, 3)
            for c in range(num_cycles):
                bursts.append(Burst(BurstType.CPU, rng.randint(15, 35)))
                if c < num_cycles - 1 and rng.random() < 0.2:
                    bursts.append(Burst(BurstType.IO, rng.randint(2, 5)))
        else:
            num_cycles = rng.randint(3, 6)
            for c in range(num_cycles):
                bursts.append(Burst(BurstType.CPU, rng.randint(2, 6)))
                if c < num_cycles - 1:
                    bursts.append(Burst(BurstType.IO, rng.randint(6, 18)))

        processes.append(Process(pid=pid, arrival_time=current_arrival, bursts=bursts))

    return processes


def generate_dynamic_workload(
    phase1_count: int = 20,
    phase2_count: int = 30,
    phase3_count: int = 25,
    seed: int = 42
) -> List[Process]:
    """
    Generates a dynamic multi-phase workload to test scheduler adaptivity:
    - Phase 1: Heavy CPU batch jobs, sparse arrivals.
    - Phase 2: Sudden arrival flash crowd of interactive, I/O-bound short tasks (workload shift).
    - Phase 3: Heterogeneous mixed background load.
    """
    rng = random.Random(seed)
    processes: List[Process] = []
    pid_counter = 0
    current_time = 0

    # Phase 1: Heavy CPU batch
    for _ in range(phase1_count):
        inter_arrival = rng.randint(4, 10)
        current_time += inter_arrival
        bursts = [
            Burst(BurstType.CPU, rng.randint(20, 45)),
            Burst(BurstType.IO, rng.randint(2, 5)),
            Burst(BurstType.CPU, rng.randint(15, 30))
        ]
        processes.append(Process(pid=pid_counter, arrival_time=current_time, bursts=bursts))
        pid_counter += 1

    # Transition to Phase 2: sudden shift to I/O flash crowd
    current_time += 10
    phase2_start_time = current_time

    for _ in range(phase2_count):
        # Bursty, rapid arrivals
        inter_arrival = rng.randint(0, 3)
        current_time += inter_arrival
        num_cycles = rng.randint(3, 6)
        bursts = []
        for c in range(num_cycles):
            bursts.append(Burst(BurstType.CPU, rng.randint(2, 5)))
            if c < num_cycles - 1:
                bursts.append(Burst(BurstType.IO, rng.randint(8, 20)))
        processes.append(Process(pid=pid_counter, arrival_time=current_time, bursts=bursts))
        pid_counter += 1

    # Transition to Phase 3: Mixed load
    current_time += 15
    for _ in range(phase3_count):
        inter_arrival = rng.randint(2, 8)
        current_time += inter_arrival
        if rng.random() < 0.5:
            # CPU job
            bursts = [
                Burst(BurstType.CPU, rng.randint(15, 30)),
                Burst(BurstType.IO, rng.randint(3, 6)),
                Burst(BurstType.CPU, rng.randint(10, 25))
            ]
        else:
            # Interactive job
            bursts = [
                Burst(BurstType.CPU, rng.randint(2, 6)),
                Burst(BurstType.IO, rng.randint(6, 16)),
                Burst(BurstType.CPU, rng.randint(2, 6))
            ]
        processes.append(Process(pid=pid_counter, arrival_time=current_time, bursts=bursts))
        pid_counter += 1

    return processes
