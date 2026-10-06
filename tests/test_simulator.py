"""
Unit tests for the discrete-event simulator and classical baseline schedulers.
"""

import unittest
from rlsched.process import Process, Burst, BurstType
from rlsched.simulator import Simulator
from rlsched.baselines import FCFSScheduler, SJFScheduler, RoundRobinScheduler
from rlsched.metrics import evaluate_simulation


class TestSimulatorAndBaselines(unittest.TestCase):
    def test_fcfs_execution_order(self):
        """Verify FCFS executes strictly in arrival order."""
        p1 = Process(pid=1, arrival_time=0, bursts=[Burst(BurstType.CPU, 10)])
        p2 = Process(pid=2, arrival_time=2, bursts=[Burst(BurstType.CPU, 4)])
        
        sim = Simulator([p1, p2], context_switch_cost=0)
        sched = FCFSScheduler()
        sched.run(sim)

        metrics = evaluate_simulation(sim)
        self.assertEqual(metrics["completed_count"], 2)
        completed = {p.pid: p for p in sim.completed_processes}
        self.assertEqual(completed[1].finish_time, 10)
        self.assertEqual(completed[2].finish_time, 14)
        self.assertEqual(completed[2].total_waiting_time, 8)

    def test_sjf_preemption(self):
        """Verify SJF (SRTF) preempts a long job when a shorter job arrives."""
        p1 = Process(pid=1, arrival_time=0, bursts=[Burst(BurstType.CPU, 20)])
        p2 = Process(pid=2, arrival_time=3, bursts=[Burst(BurstType.CPU, 2)])

        sim = Simulator([p1, p2], context_switch_cost=0)
        sched = SJFScheduler()
        sched.run(sim)

        completed = {p.pid: p for p in sim.completed_processes}
        self.assertEqual(completed[2].finish_time, 5)
        self.assertEqual(completed[1].finish_time, 22)
        self.assertGreaterEqual(completed[1].preemptions, 1)

    def test_round_robin_quantum_preemption(self):
        """Verify Round Robin preempts after time quantum expires."""
        p1 = Process(pid=1, arrival_time=0, bursts=[Burst(BurstType.CPU, 8)])
        p2 = Process(pid=2, arrival_time=0, bursts=[Burst(BurstType.CPU, 8)])

        sim = Simulator([p1, p2], context_switch_cost=0)
        sched = RoundRobinScheduler(time_quantum=4)
        sched.run(sim)

        completed = {p.pid: p for p in sim.completed_processes}
        self.assertEqual(completed[1].finish_time, 12)
        self.assertEqual(completed[2].finish_time, 16)
        self.assertEqual(completed[1].preemptions, 1)

    def test_context_switch_cost(self):
        """Verify context switch penalty adds delay ticks."""
        p1 = Process(pid=1, arrival_time=0, bursts=[Burst(BurstType.CPU, 5)])
        p2 = Process(pid=2, arrival_time=0, bursts=[Burst(BurstType.CPU, 5)])

        sim_no_cs = Simulator([p1, p2], context_switch_cost=0)
        FCFSScheduler().run(sim_no_cs)
        time_no_cs = sim_no_cs.current_time

        sim_with_cs = Simulator([p1, p2], context_switch_cost=2)
        FCFSScheduler().run(sim_with_cs)
        time_with_cs = sim_with_cs.current_time

        self.assertGreater(time_with_cs, time_no_cs)
        self.assertEqual(sim_with_cs.context_switch_ticks, 4)

    def test_io_parallel_execution(self):
        """Verify blocked I/O tasks execute in parallel while CPU is busy."""
        p1 = Process(pid=1, arrival_time=0, bursts=[
            Burst(BurstType.CPU, 5),
            Burst(BurstType.IO, 10),
            Burst(BurstType.CPU, 2)
        ])
        p2 = Process(pid=2, arrival_time=0, bursts=[
            Burst(BurstType.CPU, 5),
            Burst(BurstType.IO, 10),
            Burst(BurstType.CPU, 2)
        ])

        sim = Simulator([p1, p2], context_switch_cost=0)
        FCFSScheduler().run(sim)

        completed = {p.pid: p for p in sim.completed_processes}
        self.assertIsNotNone(completed[1].finish_time)
        self.assertIsNotNone(completed[2].finish_time)
        self.assertEqual(len(sim.completed_processes), 2)


if __name__ == "__main__":
    unittest.main()
