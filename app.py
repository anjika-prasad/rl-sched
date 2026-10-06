"""
RL-Sched — Interactive Dashboard
A Reinforcement Learning-Based Adaptive CPU Scheduler for Dynamic Workloads.

Run with:  streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rlsched.workload import (
    generate_static_cpu_bound,
    generate_static_io_bound,
    generate_static_mixed,
    generate_dynamic_workload,
)
from rlsched.simulator import Simulator
from rlsched.baselines import (
    FCFSScheduler,
    SJFScheduler,
    RoundRobinScheduler,
    RLBMCS_AdaptiveRRScheduler,
)
from rlsched.q_learning import QLearningAgent
from rlsched.rl_scheduler import RLScheduler
from rlsched.metrics import evaluate_simulation
from rlsched.visualizer import plot_q_table_heatmap, plot_gantt_chart

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RL-Sched",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  /* ── reset / base ── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding: 0 !important; max-width: 100% !important; }

  /* ── hero ── */
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f4c75 100%);
    padding: 4rem 6rem 3.5rem;
    color: white;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(16,185,129,0.18);
    color: #6ee7b7;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(110,231,183,0.35);
    margin-bottom: 1.2rem;
  }
  .hero h1 {
    font-size: 3rem;
    font-weight: 800;
    margin: 0 0 0.6rem;
    line-height: 1.15;
  }
  .hero p {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 680px;
    margin: 0;
    line-height: 1.65;
  }

  /* ── content wrapper ── */
  .content { padding: 2.5rem 6rem; }

  /* ── section heading ── */
  .section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
    margin: 2.5rem 0 1rem;
    padding-left: 0.75rem;
    border-left: 4px solid #10b981;
  }

  /* ── stat card ── */
  .stat-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }
  .stat-card {
    flex: 1 1 160px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .stat-card.winner { border: 2px solid #10b981; background: #f0fdf4; }
  .stat-card .label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #64748b; margin-bottom: 0.4rem; }
  .stat-card .value { font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
  .stat-card.winner .value { color: #059669; }
  .stat-card .meta { font-size: 0.72rem; color: #94a3b8; margin-top: 0.3rem; }

  /* ── config panel ── */
  .config-panel {
    background: #f1f5f9;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
  }
  .config-panel h3 {
    font-size: 1rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 1.2rem;
  }

  /* ── run button ── */
  div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #059669, #10b981) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 10px !important;
    border: none !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
  }
  div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

  /* ── pill tag ── */
  .pill {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
  }
  .pill-green { background: #d1fae5; color: #065f46; }
  .pill-blue  { background: #dbeafe; color: #1e40af; }
  .pill-gray  { background: #f1f5f9; color: #475569; }

  /* ── table tweaks ── */
  .stDataFrame { border-radius: 10px !important; overflow: hidden; }

  /* hide default Streamlit header/footer */
  header { visibility: hidden; }
  footer  { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD / TRAIN AGENT
# ──────────────────────────────────────────────
@st.cache_resource
def load_or_train_agent():
    model_path = os.path.join("benchmarks", "results", "trained_agent.json")
    agent = QLearningAgent(seed=42)
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        for ep in range(100):
            wl = generate_static_mixed(num_processes=15, seed=100 + ep)
            sim = Simulator(wl, context_switch_cost=1)
            sched = RLScheduler(agent, decision_interval=8, training=True)
            sched.run(sim)
            agent.decay_epsilon()
    return agent

agent = load_or_train_agent()

# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Research Project · Tabular Q-Learning</div>
  <h1>⚡ RL-Sched</h1>
  <p>An adaptive CPU scheduler that learns to tune its own time quantum and aging rate
  in real time — benchmarked against FCFS, SJF, Round Robin, and published RLBMCS.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONFIGURATION PANEL (full-width, inline)
# ──────────────────────────────────────────────
st.markdown('<div class="content">', unsafe_allow_html=True)
st.markdown('<div class="config-panel"><h3>⚙ Simulation Settings</h3>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
with c1:
    workload_type = st.selectbox(
        "Workload Profile",
        [
            "Dynamic Phase Shift",
            "Static Mixed Heterogeneous",
            "Static CPU-Bound (Batch)",
            "Static I/O-Bound (Interactive)",
        ],
        help="Dynamic Phase Shift simulates a workload that shifts character mid-run — the toughest test for static schedulers.",
    )
with c2:
    num_processes = st.slider("Processes", 10, 60, 30, 5)
with c3:
    cs_cost = st.slider("Context-Switch Cost (ticks)", 0, 3, 1)
with c4:
    decision_interval = st.slider("Decision Interval (ticks)", 4, 20, 8, 2,
        help="How often RL-Sched re-evaluates and potentially changes its scheduling parameters.")
with c5:
    seed_val = st.number_input("Seed", 1, 9999, 42, 1)

c6, c7, _ = st.columns([1, 1, 3])
with c6:
    online_mode = st.checkbox("Continuous Online Learning", value=True,
        help="RL-Sched updates its Q-table on-the-fly during execution, not just during offline training.")
with c7:
    online_lr = st.slider("Online LR", 0.01, 0.20, 0.05, 0.01) if online_mode else 0.05

st.markdown('</div>', unsafe_allow_html=True)  # config-panel

run_btn = st.button("▶  Run Simulation")

# ──────────────────────────────────────────────
# GENERATE WORKLOAD
# ──────────────────────────────────────────────
if workload_type == "Dynamic Phase Shift":
    workload = generate_dynamic_workload(
        phase1_count=max(5, num_processes // 3),
        phase2_count=max(5, num_processes // 2),
        phase3_count=max(5, num_processes // 4),
        seed=seed_val,
    )
elif workload_type == "Static Mixed Heterogeneous":
    workload = generate_static_mixed(num_processes=num_processes, seed=seed_val)
elif workload_type == "Static CPU-Bound (Batch)":
    workload = generate_static_cpu_bound(num_processes=num_processes, seed=seed_val)
else:
    workload = generate_static_io_bound(num_processes=num_processes, seed=seed_val)

# ──────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────
if run_btn:
    schedulers = [
        FCFSScheduler(),
        SJFScheduler(),
        RoundRobinScheduler(time_quantum=4),
        RLBMCS_AdaptiveRRScheduler(),
        RLScheduler(agent, decision_interval=decision_interval,
                    training=False, online_learning=online_mode, online_lr=online_lr),
    ]

    with st.spinner("Running simulation across all schedulers…"):
        results, run_records = [], {}
        for s in schedulers:
            sim = Simulator(workload, context_switch_cost=cs_cost)
            res = s.run(sim)
            m = evaluate_simulation(sim)
            m["name"] = s.name
            results.append(m)
            run_records[s.name] = {"sim": sim, "res": res}

    df = pd.DataFrame(results)
    rl_row = df[df["name"] == "RL-Sched (Proposed)"].iloc[0]
    rl_res  = run_records["RL-Sched (Proposed)"]["res"]

    # ── KPI CARDS ──
    st.markdown('<div class="section-title">Key Performance Indicators — RL-Sched vs Best Baseline</div>', unsafe_allow_html=True)

    baseline_df = df[df["name"] != "RL-Sched (Proposed)"]
    best_wait = baseline_df["avg_waiting_time"].min()
    best_fair = baseline_df["jains_fairness"].max()
    best_util = baseline_df["cpu_utilization"].max()
    best_tat  = baseline_df["avg_turnaround_time"].min()

    wait_delta = round(best_wait - rl_row["avg_waiting_time"], 1)
    fair_delta = round(rl_row["jains_fairness"] - best_fair, 3)
    util_delta = round(rl_row["cpu_utilization"] - best_util, 1)
    tat_delta  = round(best_tat - rl_row["avg_turnaround_time"], 1)

    def _delta_label(v, unit="", lower_is_better=False):
        if lower_is_better:
            if v > 0:
                return f"↓ {abs(v)}{unit} better than best baseline"
            elif v < 0:
                return f"↑ {abs(v)}{unit} worse than best baseline"
        else:
            if v > 0:
                return f"↑ {abs(v)}{unit} better than best baseline"
            elif v < 0:
                return f"↓ {abs(v)}{unit} worse than best baseline"
        return "ties best baseline"

    n_updates = rl_res.get("online_updates_count", 0)
    avg_td    = round(rl_res.get("avg_td_error", 0.0), 4)

    kpi_html = f"""
    <div class="stat-grid">
      <div class="stat-card {'winner' if wait_delta >= 0 else ''}">
        <div class="label">Avg Waiting Time</div>
        <div class="value">{round(rl_row['avg_waiting_time'],1)}</div>
        <div class="meta">ticks &nbsp;·&nbsp; {_delta_label(wait_delta,'t',True)}</div>
      </div>
      <div class="stat-card {'winner' if tat_delta >= 0 else ''}">
        <div class="label">Avg Turnaround</div>
        <div class="value">{round(rl_row['avg_turnaround_time'],1)}</div>
        <div class="meta">ticks &nbsp;·&nbsp; {_delta_label(tat_delta,'t',True)}</div>
      </div>
      <div class="stat-card {'winner' if fair_delta >= 0 else ''}">
        <div class="label">Jain's Fairness</div>
        <div class="value">{round(rl_row['jains_fairness'],3)}</div>
        <div class="meta">max 1.0 &nbsp;·&nbsp; {_delta_label(fair_delta)}</div>
      </div>
      <div class="stat-card {'winner' if util_delta >= 0 else ''}">
        <div class="label">CPU Utilization</div>
        <div class="value">{round(rl_row['cpu_utilization'],1)}%</div>
        <div class="meta">{_delta_label(util_delta,'%')}</div>
      </div>
      <div class="stat-card">
        <div class="label">Online Q-Updates</div>
        <div class="value">{n_updates}</div>
        <div class="meta">Avg TD-error: {avg_td}</div>
      </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── COMPARISON CHARTS ──
    st.markdown('<div class="section-title">Scheduler Comparison</div>', unsafe_allow_html=True)

    chart_cols = st.columns(2)

    SCHED_ORDER = [s.name for s in schedulers]
    COLOR_RULE = alt.condition(
        alt.datum.name == "RL-Sched (Proposed)",
        alt.value("#10b981"),
        alt.value("#64748b"),
    )

    with chart_cols[0]:
        c_wait = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("name:N", sort=SCHED_ORDER, title=None, axis=alt.Axis(labelAngle=-30, labelFontSize=11)),
            y=alt.Y("avg_waiting_time:Q", title="Ticks", axis=alt.Axis(grid=True, gridColor="#f1f5f9")),
            color=COLOR_RULE,
            tooltip=[alt.Tooltip("name:N", title="Scheduler"), alt.Tooltip("avg_waiting_time:Q", title="Avg Wait", format=".1f")],
        ).properties(title=alt.TitleParams("Average Waiting Time  ↓ Lower is Better", fontSize=13, fontWeight="bold"), height=280)
        st.altair_chart(c_wait, use_container_width=True)

    with chart_cols[1]:
        c_fair = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("name:N", sort=SCHED_ORDER, title=None, axis=alt.Axis(labelAngle=-30, labelFontSize=11)),
            y=alt.Y("jains_fairness:Q", scale=alt.Scale(domain=[0, 1.05]), title="Index", axis=alt.Axis(grid=True, gridColor="#f1f5f9")),
            color=COLOR_RULE,
            tooltip=[alt.Tooltip("name:N", title="Scheduler"), alt.Tooltip("jains_fairness:Q", title="Fairness", format=".3f")],
        ).properties(title=alt.TitleParams("Jain's Fairness Index  ↑ Higher is Better", fontSize=13, fontWeight="bold"), height=280)
        st.altair_chart(c_fair, use_container_width=True)

    chart_cols2 = st.columns(2)
    with chart_cols2[0]:
        c_tat = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("name:N", sort=SCHED_ORDER, title=None, axis=alt.Axis(labelAngle=-30, labelFontSize=11)),
            y=alt.Y("avg_turnaround_time:Q", title="Ticks", axis=alt.Axis(grid=True, gridColor="#f1f5f9")),
            color=COLOR_RULE,
            tooltip=[alt.Tooltip("name:N", title="Scheduler"), alt.Tooltip("avg_turnaround_time:Q", title="Avg TAT", format=".1f")],
        ).properties(title=alt.TitleParams("Avg Turnaround Time  ↓ Lower is Better", fontSize=13, fontWeight="bold"), height=280)
        st.altair_chart(c_tat, use_container_width=True)

    with chart_cols2[1]:
        c_util = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("name:N", sort=SCHED_ORDER, title=None, axis=alt.Axis(labelAngle=-30, labelFontSize=11)),
            y=alt.Y("cpu_utilization:Q", title="%", scale=alt.Scale(domain=[0, 105]), axis=alt.Axis(grid=True, gridColor="#f1f5f9")),
            color=COLOR_RULE,
            tooltip=[alt.Tooltip("name:N", title="Scheduler"), alt.Tooltip("cpu_utilization:Q", title="CPU Util %", format=".1f")],
        ).properties(title=alt.TitleParams("CPU Utilization  ↑ Higher is Better", fontSize=13, fontWeight="bold"), height=280)
        st.altair_chart(c_util, use_container_width=True)

    # ── FULL METRICS TABLE ──
    st.markdown('<div class="section-title">Full Metrics Table</div>', unsafe_allow_html=True)
    display_df = df[[
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
        "context_switch_ticks": "CS Overhead (ticks)",
    })
    styled = display_df.style \
        .highlight_min(subset=["Avg Wait (ticks)", "Avg TAT (ticks)", "Starvations", "CS Overhead (ticks)"], color="#d1fae5") \
        .highlight_max(subset=["Jain's Fairness", "CPU Util (%)", "Throughput (%/tick)"], color="#d1fae5") \
        .format({"Avg Wait (ticks)": "{:.1f}", "Avg TAT (ticks)": "{:.1f}",
                 "Jain's Fairness": "{:.3f}", "CPU Util (%)": "{:.1f}",
                 "Throughput (%/tick)": "{:.4f}"})
    st.dataframe(styled, use_container_width=True)

    # ── GANTT CHART ──
    st.markdown('<div class="section-title">Gantt Schedule Trace</div>', unsafe_allow_html=True)
    gantt_col1, gantt_col2 = st.columns([2, 1])
    with gantt_col1:
        selected_sched = st.selectbox("Scheduler to Visualise", SCHED_ORDER, index=4)
    with gantt_col2:
        sim_obj = run_records[selected_sched]["sim"]
        horizon_max = max(60, min(600, sim_obj.current_time))
        max_t = st.slider("Horizon (ticks)", 30, horizon_max, min(160, horizon_max), 10)

    sim_obj = run_records[selected_sched]["sim"]
    fig = plot_gantt_chart(sim_obj.timeline, max_ticks=max_t)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No timeline events found within the selected horizon.")

    # ── ADAPTATION HISTORY ──
    history = rl_res.get("adaptation_history", [])
    if history:
        st.markdown('<div class="section-title">RL-Sched — Dynamic Parameter Adaptation</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame(history)
        adapt_cols = st.columns(2)
        with adapt_cols[0]:
            cq = alt.Chart(df_hist).mark_line(point=True, color="#2563EB", strokeWidth=2).encode(
                x=alt.X("time:Q", title="Clock Ticks"),
                y=alt.Y("quantum:Q", title="Time Quantum (q)"),
                tooltip=["time", "quantum", "queue_len"],
            ).properties(title=alt.TitleParams("Learned Time Quantum Over Time", fontSize=13, fontWeight="bold"), height=260)
            st.altair_chart(cq, use_container_width=True)
        with adapt_cols[1]:
            ca = alt.Chart(df_hist).mark_line(point=True, color="#dc2626", strokeWidth=2).encode(
                x=alt.X("time:Q", title="Clock Ticks"),
                y=alt.Y("aging_rate:Q", title="Aging Rate (α)"),
                tooltip=["time", "aging_rate", "avg_wait"],
            ).properties(title=alt.TitleParams("Learned Aging Rate Over Time", fontSize=13, fontWeight="bold"), height=260)
            st.altair_chart(ca, use_container_width=True)

    # ── Q-TABLE HEATMAP ──
    st.markdown('<div class="section-title">Q-Table Policy Heatmap (Interpretable RL)</div>', unsafe_allow_html=True)
    fig_heat = plot_q_table_heatmap(agent)
    if fig_heat is not None:
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)

else:
    # ── LANDING PLACEHOLDER ──
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem; color: #94a3b8;">
      <div style="font-size:4rem;">⚡</div>
      <div style="font-size:1.4rem; font-weight:700; color:#1e3a5f; margin:1rem 0 0.5rem;">Configure &amp; Run</div>
      <div style="font-size:1rem;">Choose a workload profile above, adjust the simulation settings, then click <strong>▶ Run Simulation</strong>.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # content
