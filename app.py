"""
Interactive Streamlit Dashboard for RL-Sched:
A Reinforcement Learning-Based Adaptive CPU Scheduler for Dynamic Workloads.

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

# Ensure root directory is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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
from rlsched.visualizer import plot_q_table_heatmap, plot_gantt_chart, plot_dynamic_adaptation

st.set_page_config(
    page_title="RL-Sched | Adaptive CPU Scheduler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; }
    .highlight-val { font-size: 1.6rem; font-weight: 700; color: #0D9488; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_agent():
    model_path = os.path.join("benchmarks", "results", "trained_agent.json")
    agent = QLearningAgent(seed=42)
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        # Quick fallback training if not yet run
        for ep in range(100):
            wl = generate_static_mixed(num_processes=15, seed=100 + ep)
            sim = Simulator(wl, context_switch_cost=1)
            sched = RLScheduler(agent, decision_interval=8, training=True)
            sched.run(sim)
            agent.decay_epsilon()
    return agent


agent = load_or_train_agent()

# Sidebar Controls
st.sidebar.title("⚡ RL-Sched Configuration")

workload_type = st.sidebar.selectbox(
    "Workload Type",
    [
        "Dynamic Phase Shift (Recommended)",
        "Static Mixed Heterogeneous",
        "Static CPU-Bound (Batch)",
        "Static I/O-Bound (Interactive)"
    ]
)

num_processes = st.sidebar.slider("Number of Processes", min_value=10, max_value=60, value=30, step=5)
cs_cost = st.sidebar.slider("Context Switch Overhead (ticks)", min_value=0, max_value=3, value=1, step=1)
seed_val = st.sidebar.number_input("Random Workload Seed", min_value=1, max_value=9999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Agent Parameters & Learning**")
decision_interval = st.sidebar.slider("Decision Interval (ticks)", min_value=4, max_value=20, value=8, step=2)
online_mode = st.sidebar.checkbox("⚡ Continuous Online Learning", value=True, help="When enabled, RL-Sched performs O(1) Q-table updates on-the-fly during execution to adapt continuously to newly observed workloads.")
online_lr = st.sidebar.slider("Online Learning Rate", min_value=0.01, max_value=0.20, value=0.05, step=0.01) if online_mode else 0.05

st.sidebar.info("💡 **RL-Sched** learns both **Time Quantum** ($q \\in \\{2, 4, 8, 16\\}$) and **Aging Rate** ($\\alpha \\in \\{0, 0.5, 1, 2\\}$) via Tabular Q-Learning.")

# Main Title
st.markdown("<div class='main-header'>⚡ RL-Sched: Reinforcement Learning-Based Adaptive CPU Scheduler</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Comparative OS scheduling evaluation against FCFS, SJF, Round Robin, and published RLBMCS under dynamic workload shifts.</div>", unsafe_allow_html=True)

# Generate Workload
if workload_type == "Dynamic Phase Shift (Recommended)":
    workload = generate_dynamic_workload(
        phase1_count=max(5, num_processes // 3),
        phase2_count=max(5, num_processes // 2),
        phase3_count=max(5, num_processes // 4),
        seed=seed_val
    )
elif workload_type == "Static Mixed Heterogeneous":
    workload = generate_static_mixed(num_processes=num_processes, seed=seed_val)
elif workload_type == "Static CPU-Bound (Batch)":
    workload = generate_static_cpu_bound(num_processes=num_processes, seed=seed_val)
else:
    workload = generate_static_io_bound(num_processes=num_processes, seed=seed_val)

# Tabs
tab_eval, tab_gantt, tab_adapt, tab_qtable, tab_paper = st.tabs([
    "📊 Benchmark Comparison",
    "📈 Gantt Schedule Trace",
    "🔄 Dynamic Adaptation",
    "🧠 Inspect Q-Table Policy",
    "📄 Paper & Formulation"
])

# Run all 5 schedulers
schedulers = [
    FCFSScheduler(),
    SJFScheduler(),
    RoundRobinScheduler(time_quantum=4),
    RLBMCS_AdaptiveRRScheduler(),
    RLScheduler(agent, decision_interval=decision_interval, training=False, online_learning=online_mode, online_lr=online_lr)
]

with st.spinner("Executing simulation traces across all 5 schedulers..."):
    results = []
    run_records = {}
    for s in schedulers:
        sim = Simulator(workload, context_switch_cost=cs_cost)
        res = s.run(sim)
        m = evaluate_simulation(sim)
        m["name"] = s.name
        results.append(m)
        run_records[s.name] = {"sim": sim, "res": res}

df_results = pd.DataFrame(results)

with tab_eval:
    st.subheader("Performance Comparison Table")
    if online_mode:
        rl_res = run_records["RL-Sched (Proposed)"]["res"]
        n_updates = rl_res.get("online_updates_count", 0)
        avg_td = rl_res.get("avg_td_error", 0.0)
        st.success(f"🟢 **Continuous Online Learning Active**: RL-Sched executed **{n_updates} on-the-fly Q-table updates** during this simulation run (Avg TD-Error: `{avg_td}`). The Q-table policy continuously adapted to the observed trace in $O(1)$ time.")

    display_df = df_results[[
        "name", "avg_waiting_time", "avg_turnaround_time", "jains_fairness",
        "throughput", "cpu_utilization", "starvation_count", "context_switch_ticks"
    ]].rename(columns={
        "name": "Scheduler",
        "avg_waiting_time": "Avg Wait (ticks)",
        "avg_turnaround_time": "Avg TAT (ticks)",
        "jains_fairness": "Jain's Fairness",
        "throughput": "Throughput (%/tick)",
        "cpu_utilization": "CPU Util (%)",
        "starvation_count": "Starvations",
        "context_switch_ticks": "CS Overhead (ticks)"
    })

    st.dataframe(display_df.style.highlight_min(
        subset=["Avg Wait (ticks)", "Avg TAT (ticks)", "Starvations", "CS Overhead (ticks)"],
        color="#d1fae5"
    ).highlight_max(
        subset=["Jain's Fairness", "CPU Util (%)", "Throughput (%/tick)"],
        color="#d1fae5"
    ), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average Waiting Time (Lower is Better)")
        chart_awt = alt.Chart(df_results).mark_bar().encode(
            x=alt.X("name:N", sort=None, title="Scheduler"),
            y=alt.Y("avg_waiting_time:Q", title="Ticks"),
            color=alt.condition(
                alt.datum.name == "RL-Sched (Proposed)",
                alt.value("#10B981"),
                alt.value("#6B7280")
            ),
            tooltip=["name", "avg_waiting_time"]
        ).properties(height=320)
        st.altair_chart(chart_awt, use_container_width=True)

    with col2:
        st.markdown("#### Jain's Fairness Index (Higher is Better, Max 1.0)")
        chart_fair = alt.Chart(df_results).mark_bar().encode(
            x=alt.X("name:N", sort=None, title="Scheduler"),
            y=alt.Y("jains_fairness:Q", scale=alt.Scale(domain=[0, 1.05]), title="Index"),
            color=alt.condition(
                alt.datum.name == "RL-Sched (Proposed)",
                alt.value("#10B981"),
                alt.value("#6B7280")
            ),
            tooltip=["name", "jains_fairness"]
        ).properties(height=320)
        st.altair_chart(chart_fair, use_container_width=True)

with tab_gantt:
    st.subheader("Process CPU Execution Timeline (Gantt Chart)")
    selected_sched = st.selectbox("Select Scheduler to Inspect", [s.name for s in schedulers], index=4)
    sim_obj = run_records[selected_sched]["sim"]
    max_t = st.slider("Timeline Horizon (Clock Ticks)", min_value=50, max_value=min(600, sim_obj.current_time), value=160, step=10)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    plot_gantt_chart(sim_obj.timeline, max_ticks=max_t)
    st.pyplot(fig)
    plt.close()

with tab_adapt:
    st.subheader("Dynamic Parameter Adaptation History")
    st.markdown("Shows how RL-Sched continuously reacts to system state by adjusting Time Quantum ($q$) and Aging Rate ($\\alpha$).")
    rl_run = run_records["RL-Sched (Proposed)"]["res"]
    history = rl_run.get("adaptation_history", [])

    if history:
        df_hist = pd.DataFrame(history)
        col_q, col_a = st.columns(2)
        with col_q:
            chart_q = alt.Chart(df_hist).mark_line(point=True, color="#2563EB").encode(
                x=alt.X("time:Q", title="Clock Ticks"),
                y=alt.Y("quantum:Q", title="Time Quantum (q)"),
                tooltip=["time", "quantum", "queue_len"]
            ).properties(title="Learned Time Quantum (q) Over Time", height=280)
            st.altair_chart(chart_q, use_container_width=True)

        with col_a:
            chart_a = alt.Chart(df_hist).mark_line(point=True, color="#DC2626").encode(
                x=alt.X("time:Q", title="Clock Ticks"),
                y=alt.Y("aging_rate:Q", title="Aging Rate (alpha)"),
                tooltip=["time", "aging_rate", "avg_wait"]
            ).properties(title="Learned Aging Rate (alpha) Over Time", height=280)
            st.altair_chart(chart_a, use_container_width=True)
    else:
        st.info("No adaptation steps logged.")

with tab_qtable:
    st.subheader("Tabular Q-Learning Policy Heatmap & State Inspector")
    st.markdown("Unlike black-box Deep RL, RL-Sched's tabular Q-table is **100% transparent and inspectable**.")
    fig_heat, ax_heat = plt.subplots(figsize=(8, 5))
    plot_q_table_heatmap(agent)
    st.pyplot(fig_heat)
    plt.close()

    st.markdown("#### State Lookup & Action Decoder")
    test_cpu = st.selectbox("Observed CPU Utilization", ["Low (<40%)", "Med (40-80%)", "High (>80%)"], index=1)
    test_qlen = st.selectbox("Ready Queue Length", ["Short (<=2)", "Med (3-6)", "Long (>6)"], index=1)
    test_burst = st.selectbox("Estimated Burst Length", ["Short (<6)", "Med (6-15)", "Long (>15)"], index=1)
    test_io = st.selectbox("I/O-to-CPU Ratio", ["Low (<25%)", "Balanced (25-60%)", "High (>60%)"], index=1)

    c_idx = ["Low (<40%)", "Med (40-80%)", "High (>80%)"].index(test_cpu)
    q_idx = ["Short (<=2)", "Med (3-6)", "Long (>6)"].index(test_qlen)
    b_idx = ["Short (<6)", "Med (6-15)", "Long (>15)"].index(test_burst)
    io_idx = ["Low (<25%)", "Balanced (25-60%)", "High (>60%)"].index(test_io)

    s_idx = c_idx * 27 + q_idx * 9 + b_idx * 3 + io_idx
    best_a = int(np.argmax(agent.q_table[s_idx]))
    opt_q, opt_alpha = agent.get_action_params(best_a)

    st.success(f"**State #{s_idx} Policy**: Optimal Time Quantum $q = {opt_q}$ ticks, Aging Rate $\\alpha = {opt_alpha}$")

with tab_paper:
    st.subheader("Conference Paper Positioning & Mathematical Formulation")
    st.markdown(r"""
    ### 1. MDP Formulation
    - **State Space $S$**: $\langle \text{CPU Util}, \text{Queue Length}, \text{Burst Estimate}, \text{I/O Ratio} \rangle \in [0, 80]$ (81 discretized states).
    - **Action Space $A$**: Coupled discrete tuples $(q, \alpha) \in \{2, 4, 8, 16\} \times \{0.0, 0.5, 1.0, 2.0\}$ (16 discrete actions).
    - **Composite Reward Function $R$**:
      $$R = - w_1 \cdot \overline{W}_{\text{norm}} + w_2 \cdot \text{Throughput}_{\text{norm}} + w_3 \cdot \text{Jain's Fairness} - w_4 \cdot \text{Starvation}$$
    
    ### 2. Comparison with Published Baselines
    - **FCFS**: Prone to convoy effect when batch tasks arrive ahead of interactive tasks.
    - **SJF (SRTF)**: Achieves low waiting times at the cost of catastrophic starvation for long jobs.
    - **Round Robin**: Fixed time quantum causes high context-switch overhead or convoy delays.
    - **RLBMCS / Dyn-RR (Rinku et al. & Shrivastava 2010)**: Published single-parameter adaptation that tunes only time quantum without dynamic aging or multi-objective fairness feedback.
    - **RL-Sched (This Project)**: Jointly adapts both time quantum and aging rate, achieving lower turnaround times than Round Robin and RLBMCS while maintaining strict zero-starvation fairness.
    """)
