"""
RL-Sched Scheduler:
Couples the discrete-event simulator with the Tabular Q-Learning Agent.
Jointly adapts time quantum (q) and aging rate (alpha) based on observed system telemetry.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .simulator import Simulator
from .process import Process, ProcessState, BurstType
from .q_learning import QLearningAgent
from .baselines import BaseScheduler
from .metrics import calculate_jains_fairness_index


class RLScheduler(BaseScheduler):
    """
    RL-Sched: Adaptive Reinforcement Learning-based CPU Scheduler.
    Learns to dynamically adjust (time_quantum, aging_rate) pairs.
    """
    def __init__(
        self,
        agent: QLearningAgent,
        decision_interval: int = 8,
        training: bool = False,
        online_learning: bool = False,
        online_lr: float = 0.05,
        online_epsilon: float = 0.02,
        w_wait: float = 1.0,
        w_throughput: float = 1.2,
        w_fairness: float = 1.5,
        w_starve: float = 1.0
    ):
        super().__init__(name="RL-Sched (Proposed)")
        self.agent = agent
        self.decision_interval = decision_interval
        self.training = training
        self.online_learning = online_learning
        self.online_lr = online_lr
        self.online_epsilon = online_epsilon

        # Multi-objective reward weights
        self.w_wait = w_wait
        self.w_throughput = w_throughput
        self.w_fairness = w_fairness
        self.w_starve = w_starve

        # Active parameters
        self.current_quantum: int = 4
        self.current_aging_rate: float = 0.5
        self.current_action_idx: int = 0

        # Adaptation and online updates tracking
        self.adaptation_history: List[Dict[str, Any]] = []
        self.online_updates_count: int = 0
        self.total_td_error: float = 0.0

    def compute_reward(self, sim: Simulator, telemetry: Dict[str, float]) -> float:
        """
        Computes composite multi-objective reward balancing:
        - Low waiting time
        - High throughput
        - High Jain's fairness index
        - Starvation penalty
        """
        avg_wait = telemetry.get("avg_current_wait", 0.0)
        norm_wait_penalty = min(2.5, avg_wait / 15.0)

        tp = telemetry.get("throughput", 0.0)
        tp_reward = min(2.0, tp * 5.0)

        wait_times = [sim.current_time - p.last_ready_time for p in sim.ready_queue]
        if wait_times and len(wait_times) > 1:
            fairness = calculate_jains_fairness_index(wait_times)
        else:
            fairness = 1.0

        max_wait = telemetry.get("max_current_wait", 0.0)
        starve_penalty = 1.5 if max_wait > 50 else (0.5 if max_wait > 25 else 0.0)

        reward = (
            - self.w_wait * norm_wait_penalty
            + self.w_throughput * tp_reward
            + self.w_fairness * fairness
            - self.w_starve * starve_penalty
        )
        self.last_parts = {
            "wait": -self.w_wait * norm_wait_penalty,
            "throughput": self.w_throughput * tp_reward,
            "fairness": self.w_fairness * fairness,
            "starvation": -self.w_starve * starve_penalty,
        }
        return float(reward)

    def apply_aging_and_sort_queue(self, sim: Simulator):
        """
        Applies current aging rate to all ready processes and sorts the queue
        by dynamic priority (descending).
        """
        for p in sim.ready_queue:
            p.update_dynamic_priority(self.current_aging_rate, sim.current_time)

        sim.ready_queue.sort(
            key=lambda p: (-p.dynamic_priority, p.remaining_cpu_time, p.arrival_time)
        )

    def run(self, sim: Simulator) -> Dict[str, Any]:
        quantum_spent = 0
        ticks_since_decision = 0
        last_state: Optional[int] = None
        last_action: Optional[int] = None
        self.adaptation_history.clear()

        while not sim.is_done() and sim.current_time < sim.max_simulation_ticks:
            sim.admit_new_arrivals()
            sim.step_io()

            # RL Decision Point
            if ticks_since_decision >= self.decision_interval or sim.running_process is None:
                telemetry = sim.get_system_telemetry()
                current_state = self.agent.discretize_state(telemetry)
                # Check if learning (either offline training or live online learning)
                should_update = (self.training or self.online_learning) and last_state is not None and last_action is not None
                if should_update:
                    step_reward = self.compute_reward(sim, telemetry)
                    active_lr = self.online_lr if self.online_learning else None
                    td_err = self.agent.update(last_state, last_action, step_reward, current_state, custom_lr=active_lr)
                    if self.online_learning:
                        self.online_updates_count += 1
                        self.total_td_error += td_err

                # Action selection:
                # - Training: uses decaying agent.epsilon
                # - Online learning: uses low background online_epsilon
                # - Pure evaluation: 100% greedy (evaluate=True)
                if self.training:
                    action_idx = self.agent.select_action(current_state, evaluate=False)
                elif self.online_learning:
                    if self.agent.rng.random() < self.online_epsilon:
                        action_idx = self.agent.rng.randint(0, self.agent.num_actions - 1)
                    else:
                        action_idx = self.agent.select_action(current_state, evaluate=True)
                else:
                    action_idx = self.agent.select_action(current_state, evaluate=True)

                new_quantum, new_aging = self.agent.get_action_params(action_idx)

                self.current_action_idx = action_idx
                self.current_quantum = new_quantum
                self.current_aging_rate = new_aging
                last_state = current_state
                last_action = action_idx
                ticks_since_decision = 0

                self.apply_aging_and_sort_queue(sim)

                self.adaptation_history.append({
                    "time": sim.current_time,
                    "quantum": new_quantum,
                    "aging_rate": new_aging,
                    "state": current_state,
                    "cpu_util": telemetry["cpu_utilization"],
                    "queue_len": telemetry["ready_queue_length"],
                    "avg_wait": telemetry["avg_current_wait"],
                    "q_values": self.agent.q_table[current_state].tolist(),
                    "reward_parts": dict(getattr(self, "last_parts", {}))
                })

            ticks_since_decision += 1

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
                    quantum_spent = 0
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
                    self.apply_aging_and_sort_queue(sim)
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

            # Execute running process
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

        if (self.training or self.online_learning) and last_state is not None and last_action is not None:
            final_telemetry = sim.get_system_telemetry()
            final_state = self.agent.discretize_state(final_telemetry)
            final_reward = self.compute_reward(sim, final_telemetry)
            active_lr = self.online_lr if self.online_learning else None
            td_err = self.agent.update(last_state, last_action, final_reward, final_state, custom_lr=active_lr)
            if self.online_learning:
                self.online_updates_count += 1
                self.total_td_error += td_err

        avg_td = (self.total_td_error / self.online_updates_count) if self.online_updates_count > 0 else 0.0

        return {
            "name": self.name,
            "simulator": sim,
            "adaptation_history": self.adaptation_history,
            "online_updates_count": self.online_updates_count,
            "avg_td_error": round(avg_td, 4)
        }
