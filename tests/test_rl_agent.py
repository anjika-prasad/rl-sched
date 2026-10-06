"""
Unit tests for the tabular Q-learning agent and state discretization.
"""

import os
import tempfile
import unittest
import numpy as np
from rlsched.q_learning import QLearningAgent
from rlsched.workload import generate_static_mixed
from rlsched.simulator import Simulator
from rlsched.rl_scheduler import RLScheduler


class TestRLAgent(unittest.TestCase):
    def test_q_table_dimensions(self):
        agent = QLearningAgent()
        self.assertEqual(agent.q_table.shape, (81, 16))
        self.assertEqual(len(agent.actions), 16)

    def test_state_discretization_coverage(self):
        agent = QLearningAgent()
        
        # Low everything
        s1 = agent.discretize_state({"cpu_utilization": 0.1, "ready_queue_length": 1, "avg_burst_estimate": 2.0, "io_ratio": 0.1})
        self.assertTrue(0 <= s1 < 81)

        # High everything
        s2 = agent.discretize_state({"cpu_utilization": 0.95, "ready_queue_length": 10, "avg_burst_estimate": 25.0, "io_ratio": 0.8})
        self.assertEqual(s2, 80)  # 2*27 + 2*9 + 2*3 + 2 = 54 + 18 + 6 + 2 = 80

    def test_q_learning_bellman_update(self):
        agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9, epsilon=0.0)
        state = 10
        action = 2
        reward = 5.0
        next_state = 12

        # Set known Q value for next_state
        agent.q_table[next_state, :] = 2.0

        old_q = agent.q_table[state, action]
        agent.update(state, action, reward, next_state)
        new_q = agent.q_table[state, action]

        # Target = reward + gamma * max_Q(next_state) = 5.0 + 0.9 * 2.0 = 6.8
        # new_q = old_q + lr * (target - old_q) = 0 + 0.5 * 6.8 = 3.4
        self.assertTrue(np.isclose(new_q, 3.4))

    def test_save_and_load_agent(self):
        agent = QLearningAgent()
        agent.q_table[5, 3] = 42.5
        agent.epsilon = 0.25

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            agent.save(tmp_path)
            new_agent = QLearningAgent()
            new_agent.load(tmp_path)

            self.assertTrue(np.isclose(new_agent.q_table[5, 3], 42.5))
            self.assertTrue(np.isclose(new_agent.epsilon, 0.25))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_rl_scheduler_execution_loop(self):
        """Verify RL scheduler runs without crashing and completes processes."""
        workload = generate_static_mixed(num_processes=10, seed=123)
        sim = Simulator(workload, context_switch_cost=1)
        agent = QLearningAgent(epsilon=0.2)
        scheduler = RLScheduler(agent, decision_interval=5, training=True)

        result = scheduler.run(sim)
        self.assertEqual(len(sim.completed_processes), 10)
        self.assertGreater(len(result["adaptation_history"]), 0)

    def test_online_learning_updates(self):
        """Verify scheduler in online_learning mode updates Q-table on-the-fly during execution."""
        workload = generate_static_mixed(num_processes=10, seed=456)
        sim = Simulator(workload, context_switch_cost=1)
        agent = QLearningAgent(epsilon=0.0)
        initial_q_sum = float(np.sum(np.abs(agent.q_table)))

        scheduler = RLScheduler(agent, decision_interval=4, training=False, online_learning=True)
        res = scheduler.run(sim)

        self.assertGreater(res["online_updates_count"], 0)
        new_q_sum = float(np.sum(np.abs(agent.q_table)))
        # Q-table must have been updated during execution
        self.assertGreater(new_q_sum, initial_q_sum)


if __name__ == "__main__":
    unittest.main()
