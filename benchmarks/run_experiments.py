"""
Automated Benchmarking Suite for RL-Sched:
Trains the Q-learning agent and performs comprehensive comparative evaluation
against FCFS, SJF, Round Robin, and RLBMCS (Rinku et al. 2021 / Shrivastava 2010 published baseline).
Generates publication-quality charts and CSV summaries in benchmarks/results/.
"""

import os
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from rlsched.workload import (
    generate_static_cpu_bound,
    generate_static_io_bound,
    generate_static_mixed,
    generate_dynamic_workload
)
from rlsched.simulator import Simulator
from rlsched.baselines import (
    FCFSScheduler,
    SJFScheduler,
    RoundRobinScheduler,
    RLBMCS_AdaptiveRRScheduler
)
from rlsched.q_learning import QLearningAgent
from rlsched.rl_scheduler import RLScheduler
from rlsched.metrics import evaluate_simulation
from rlsched.visualizer import (
    plot_scheduler_comparison,
    plot_dynamic_adaptation,
    plot_q_table_heatmap,
    plot_gantt_chart
)


def ensure_output_dir(path: str = "benchmarks/results") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def train_rl_agent(num_episodes: int = 250, seed: int = 100) -> QLearningAgent:
    """
    Trains RL-Sched's Tabular Q-Learning Agent across varied workloads.
    """
    print("=" * 70)
    print(f"[*] Training RL-Sched Tabular Q-Learning Agent ({num_episodes} episodes)...")
    print("=" * 70)

    agent = QLearningAgent(
        learning_rate=0.15,
        discount_factor=0.90,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985,
        seed=seed
    )

    start_time = time.time()
    for ep in range(1, num_episodes + 1):
        # Rotate through workload varieties during training
        scenario_choice = ep % 4
        ep_seed = seed + ep * 13

        if scenario_choice == 0:
            workload = generate_static_cpu_bound(num_processes=20, seed=ep_seed)
        elif scenario_choice == 1:
            workload = generate_static_io_bound(num_processes=20, seed=ep_seed)
        elif scenario_choice == 2:
            workload = generate_static_mixed(num_processes=25, seed=ep_seed)
        else:
            workload = generate_dynamic_workload(phase1_count=12, phase2_count=18, phase3_count=12, seed=ep_seed)

        sim = Simulator(workload, context_switch_cost=1)
        scheduler = RLScheduler(agent, decision_interval=8, training=True)
        scheduler.run(sim)
        agent.decay_epsilon()

        if ep % 50 == 0 or ep == num_episodes:
            non_zero_q = np.count_nonzero(agent.q_table)
            print(f"  Episode {ep:3d}/{num_episodes} | Epsilon: {agent.epsilon:.4f} | Non-zero Q-entries: {non_zero_q}/{agent.num_states * agent.num_actions}")

    elapsed = time.time() - start_time
    print(f"[+] Training completed in {elapsed:.2f} seconds.")
    return agent


def run_benchmark_scenario(
    workload_name: str,
    workload: list,
    agent: QLearningAgent,
    context_switch_cost: int = 1
) -> tuple[pd.DataFrame, dict]:
    """
    Runs all 5 schedulers on the exact same workload trace.
    """
    schedulers = [
        FCFSScheduler(),
        SJFScheduler(),
        RoundRobinScheduler(time_quantum=4),
        RLBMCS_AdaptiveRRScheduler(),
        RLScheduler(agent, decision_interval=8, training=False)
    ]

    scenario_metrics = []
    run_objects = {}

    for sched in schedulers:
        # Clone clean simulator instance
        sim = Simulator(workload, context_switch_cost=context_switch_cost)
        res = sched.run(sim)
        metrics = evaluate_simulation(sim)
        metrics["scheduler"] = sched.name
        metrics["name"] = sched.name
        scenario_metrics.append(metrics)
        run_objects[sched.name] = {
            "simulator": sim,
            "metrics": metrics,
            "res": res
        }

    df = pd.DataFrame(scenario_metrics)
    return df, run_objects


def main():
    out_dir = ensure_output_dir("benchmarks/results")

    # Step 1: Train RL Agent
    agent = train_rl_agent(num_episodes=250, seed=42)
    agent_save_path = os.path.join(out_dir, "trained_agent.json")
    agent.save(agent_save_path)
    print(f"[+] Saved trained Q-table agent to {agent_save_path}")

    # Plot Q-Table Policy Heatmap
    heatmap_path = os.path.join(out_dir, "learned_policy_heatmap.png")
    plot_q_table_heatmap(agent, save_path=heatmap_path)
    print(f"[+] Exported Q-table policy heatmap to {heatmap_path}")

    # Step 2: Evaluation Scenarios
    eval_seeds = {"cpu": 901, "io": 902, "mixed": 903, "dynamic": 904}
    all_summary_rows = []

    # Scenario 1: Static CPU-Bound
    print("\n" + "=" * 70)
    print("[*] Running Scenario 1: Static CPU-Bound Workload")
    print("=" * 70)
    cpu_workload = generate_static_cpu_bound(num_processes=35, seed=eval_seeds["cpu"])
    df_cpu, _ = run_benchmark_scenario("Static CPU-Bound", cpu_workload, agent)
    df_cpu["workload"] = "Static CPU-Bound"
    all_summary_rows.append(df_cpu)
    print(df_cpu[["name", "avg_waiting_time", "avg_turnaround_time", "jains_fairness", "starvation_count"]].to_string(index=False))
    plot_scheduler_comparison(df_cpu.to_dict("records"), "Static CPU-Bound Workload", os.path.join(out_dir, "static_cpu_comparison.png"))

    # Scenario 2: Static I/O-Bound
    print("\n" + "=" * 70)
    print("[*] Running Scenario 2: Static I/O-Bound Interactive Workload")
    print("=" * 70)
    io_workload = generate_static_io_bound(num_processes=35, seed=eval_seeds["io"])
    df_io, _ = run_benchmark_scenario("Static I/O-Bound", io_workload, agent)
    df_io["workload"] = "Static I/O-Bound"
    all_summary_rows.append(df_io)
    print(df_io[["name", "avg_waiting_time", "avg_turnaround_time", "jains_fairness", "starvation_count"]].to_string(index=False))
    plot_scheduler_comparison(df_io.to_dict("records"), "Static I/O-Bound Workload", os.path.join(out_dir, "static_io_comparison.png"))

    # Scenario 3: Static Mixed
    print("\n" + "=" * 70)
    print("[*] Running Scenario 3: Heterogeneous Mixed Workload")
    print("=" * 70)
    mixed_workload = generate_static_mixed(num_processes=40, seed=eval_seeds["mixed"])
    df_mixed, _ = run_benchmark_scenario("Static Mixed", mixed_workload, agent)
    df_mixed["workload"] = "Static Mixed"
    all_summary_rows.append(df_mixed)
    print(df_mixed[["name", "avg_waiting_time", "avg_turnaround_time", "jains_fairness", "starvation_count"]].to_string(index=False))
    plot_scheduler_comparison(df_mixed.to_dict("records"), "Static Mixed Workload", os.path.join(out_dir, "static_mixed_comparison.png"))

    # Scenario 4: Dynamic Shifting Workload (Phase 1 CPU -> Phase 2 IO Flash Crowd -> Phase 3 Mixed)
    print("\n" + "=" * 70)
    print("[*] Running Scenario 4: Dynamic Workload Phase Shift (Adaptation Evaluation)")
    print("=" * 70)
    dynamic_workload = generate_dynamic_workload(phase1_count=20, phase2_count=30, phase3_count=20, seed=eval_seeds["dynamic"])
    df_dyn, dyn_runs = run_benchmark_scenario("Dynamic Workload Shift", dynamic_workload, agent)
    df_dyn["workload"] = "Dynamic Shift"
    all_summary_rows.append(df_dyn)
    print(df_dyn[["name", "avg_waiting_time", "avg_turnaround_time", "jains_fairness", "starvation_count"]].to_string(index=False))
    plot_scheduler_comparison(df_dyn.to_dict("records"), "Dynamic Workload Phase Shift", os.path.join(out_dir, "dynamic_workload_comparison.png"))

    # Dynamic Adaptation Plot for RL-Sched
    rl_run = dyn_runs["RL-Sched (Proposed)"]
    adaptation_history = rl_run["res"].get("adaptation_history", [])
    adaptation_plot_path = os.path.join(out_dir, "dynamic_adaptation_curve.png")
    # Identify shift times roughly from workload
    plot_dynamic_adaptation(adaptation_history, shift_ticks=[dynamic_workload[20].arrival_time, dynamic_workload[50].arrival_time], save_path=adaptation_plot_path)
    print(f"[+] Exported dynamic adaptation curve to {adaptation_plot_path}")

    # Gantt Chart for RL-Sched Execution
    gantt_path = os.path.join(out_dir, "dynamic_gantt_chart.png")
    plot_gantt_chart(rl_run["simulator"].timeline, max_ticks=160, save_path=gantt_path)
    print(f"[+] Exported simulation Gantt chart to {gantt_path}")

    # Step 3: Export Master CSV
    master_df = pd.concat(all_summary_rows, ignore_index=True)
    master_csv_path = os.path.join(out_dir, "benchmark_summary.csv")
    master_df.to_csv(master_csv_path, index=False)
    print("\n" + "=" * 70)
    print(f"[+] Master benchmark results saved to {master_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
