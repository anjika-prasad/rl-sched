"""
Tabular Q-Learning Agent for RL-Sched.
Discretizes continuous system state into an interpretable state space (81 states)
and selects coupled (quantum, aging_rate) actions (16 actions).
"""

import json
import random
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning Agent with state discretization and epsilon-greedy exploration.
    """
    QUANTUM_OPTIONS = [2, 4, 8, 16]
    AGING_OPTIONS = [0.0, 0.5, 1.0, 2.0]

    def __init__(
        self,
        learning_rate: float = 0.15,
        discount_factor: float = 0.90,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Build action space: 16 discrete pairs (quantum, aging_rate)
        self.actions: List[Tuple[int, float]] = []
        for q in self.QUANTUM_OPTIONS:
            for alpha in self.AGING_OPTIONS:
                self.actions.append((q, alpha))
        self.num_actions = len(self.actions)  # 16

        # State space: 3 * 3 * 3 * 3 = 81 states
        self.num_states = 81
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

    def discretize_state(self, telemetry: Dict[str, float]) -> int:
        """
        Discretizes system telemetry into a single state index in [0, 80].
        Features:
        1. CPU utilization: Low (< 0.4), Med (0.4 - 0.8), High (> 0.8)
        2. Ready queue length: Low (<= 2), Med (3 - 6), High (> 6)
        3. Avg burst estimate: Short (< 6), Med (6 - 15), Long (> 15)
        4. I/O ratio: Low (< 0.25), Balanced (0.25 - 0.6), High (> 0.6)
        """
        # Feature 1: CPU Util
        cpu = telemetry.get("cpu_utilization", 0.0)
        if cpu < 0.40:
            bin_cpu = 0
        elif cpu <= 0.80:
            bin_cpu = 1
        else:
            bin_cpu = 2

        # Feature 2: Ready Queue Length
        qlen = telemetry.get("ready_queue_length", 0)
        if qlen <= 2:
            bin_qlen = 0
        elif qlen <= 6:
            bin_qlen = 1
        else:
            bin_qlen = 2

        # Feature 3: Avg Burst Estimate
        burst = telemetry.get("avg_burst_estimate", 0.0)
        if burst < 6.0:
            bin_burst = 0
        elif burst <= 15.0:
            bin_burst = 1
        else:
            bin_burst = 2

        # Feature 4: I/O Ratio
        io_ratio = telemetry.get("io_ratio", 0.0)
        if io_ratio < 0.25:
            bin_io = 0
        elif io_ratio <= 0.60:
            bin_io = 1
        else:
            bin_io = 2

        state_idx = bin_cpu * 27 + bin_qlen * 9 + bin_burst * 3 + bin_io
        return state_idx

    def get_state_labels(self, state_idx: int) -> Dict[str, str]:
        """Reverse map state index to human-readable labels for interpretability."""
        bin_cpu = state_idx // 27
        rem1 = state_idx % 27
        bin_qlen = rem1 // 9
        rem2 = rem1 % 9
        bin_burst = rem2 // 3
        bin_io = rem2 % 3

        cpu_str = ["Low CPU (<40%)", "Med CPU (40-80%)", "High CPU (>80%)"][bin_cpu]
        q_str = ["Short Queue (<=2)", "Med Queue (3-6)", "Long Queue (>6)"][bin_qlen]
        b_str = ["Short Bursts (<6)", "Med Bursts (6-15)", "Long Bursts (>15)"][bin_burst]
        io_str = ["Low I/O (<25%)", "Balanced I/O (25-60%)", "High I/O (>60%)"][bin_io]

        return {
            "cpu": cpu_str,
            "queue": q_str,
            "burst": b_str,
            "io": io_str
        }

    def select_action(self, state_idx: int, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        If evaluate=True, chooses the best action deterministically (argmax).
        """
        if not evaluate and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.num_actions - 1)
        
        # Argmax with tie-breaking
        q_values = self.q_table[state_idx]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, custom_lr: Optional[float] = None) -> float:
        """
        Standard Q-learning (Bellman) update:
        Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        lr_to_use = custom_lr if custom_lr is not None else self.lr
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += lr_to_use * td_error
        return float(abs(td_error))

    def decay_epsilon(self):
        """Decay exploration rate after an episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_action_params(self, action_idx: int) -> Tuple[int, float]:
        """Returns (quantum, aging_rate) for the given action index."""
        return self.actions[action_idx]

    def save(self, filepath: str):
        """Saves Q-table and agent configuration to disk."""
        data = {
            "q_table": self.q_table.tolist(),
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "actions": self.actions
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Loads Q-table and configuration from disk."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.q_table = np.array(data["q_table"], dtype=float)
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.lr = data.get("lr", self.lr)
        self.gamma = data.get("gamma", self.gamma)
