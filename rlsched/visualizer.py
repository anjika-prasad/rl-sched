"""
Publication-grade visualizer for RL-Sched:
- Multi-metric benchmark comparisons
- Dynamic adaptation timeline curves
- Q-table policy interpretability heatmaps
- Gantt schedule charts
"""

from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def set_paper_style():
    """Sets a clean, publication-grade aesthetics style."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300
    })


def plot_scheduler_comparison(
    results_list: List[Dict[str, Any]],
    workload_title: str = "Benchmark Evaluation",
    save_path: Optional[str] = None
):
    """
    Plots a 4-panel comparison bar chart across all schedulers:
    1. Average Waiting Time (AWT)
    2. Average Turnaround Time (ATT)
    3. Jain's Fairness Index
    4. Starvation Count & Preemptions
    """
    set_paper_style()
    df = pd.DataFrame(results_list)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Performance Comparison: {workload_title}", fontweight="bold", y=0.98)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    # If RL-Sched is in the list, highlight it
    palette = []
    for name in df["name"]:
        if "RL-Sched" in name:
            palette.append("#2ca02c")  # Highlight green
        elif "RLBMCS" in name:
            palette.append("#ff7f0e")  # Literature baseline orange
        elif "SJF" in name:
            palette.append("#1f77b4")
        elif "Round Robin" in name:
            palette.append("#9467bd")
        else:
            palette.append("#7f7f7f")

    # 1. Average Waiting Time
    ax1 = axes[0, 0]
    sns.barplot(x="name", y="avg_waiting_time", data=df, ax=ax1, palette=palette, hue="name", legend=False, edgecolor="black", linewidth=0.8)
    ax1.set_title("Average Waiting Time (Lower is Better)")
    ax1.set_xlabel("")
    ax1.set_ylabel("Ticks")
    ax1.tick_params(axis="x", rotation=15)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # 2. Average Turnaround Time
    ax2 = axes[0, 1]
    sns.barplot(x="name", y="avg_turnaround_time", data=df, ax=ax2, palette=palette, hue="name", legend=False, edgecolor="black", linewidth=0.8)
    ax2.set_title("Average Turnaround Time (Lower is Better)")
    ax2.set_xlabel("")
    ax2.set_ylabel("Ticks")
    ax2.tick_params(axis="x", rotation=15)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # 3. Jain's Fairness Index
    ax3 = axes[1, 0]
    sns.barplot(x="name", y="jains_fairness", data=df, ax=ax3, palette=palette, hue="name", legend=False, edgecolor="black", linewidth=0.8)
    ax3.set_title("Jain's Fairness Index (Higher is Better, Max=1.0)")
    ax3.set_xlabel("")
    ax3.set_ylabel("Index [0 - 1.0]")
    ax3.set_ylim(0, 1.1)
    ax3.tick_params(axis="x", rotation=15)
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # 4. Starvation Count
    ax4 = axes[1, 1]
    sns.barplot(x="name", y="starvation_count", data=df, ax=ax4, palette=palette, hue="name", legend=False, edgecolor="black", linewidth=0.8)
    ax4.set_title("Starvation Count (Lower is Better)")
    ax4.set_xlabel("")
    ax4.set_ylabel("Starved Processes")
    ax4.tick_params(axis="x", rotation=15)
    for p in ax4.patches:
        ax4.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_dynamic_adaptation(
    adaptation_history: List[Dict[str, Any]],
    shift_ticks: List[int] = None,
    save_path: Optional[str] = None
):
    """
    Plots the learned adaptation trajectory over time during workload phase shifts.
    Shows the coupling between Time Quantum (q) and Aging Rate (alpha).
    """
    set_paper_style()
    df = pd.DataFrame(adaptation_history)
    if df.empty:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("RL-Sched Parameter Adaptation Across Workload Phase Shifts", fontweight="bold")

    # 1. Quantum Adaptation
    ax1.plot(df["time"], df["quantum"], color="#1f77b4", linewidth=2.0, marker="o", markersize=3, label="Time Quantum (q)")
    ax1.set_ylabel("Quantum (ticks)")
    ax1.set_yticks([2, 4, 8, 16])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # 2. Aging Rate Adaptation
    ax2.plot(df["time"], df["aging_rate"], color="#d62728", linewidth=2.0, marker="s", markersize=3, label="Aging Rate (alpha)")
    ax2.set_ylabel("Aging Rate")
    ax2.set_yticks([0.0, 0.5, 1.0, 2.0])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    # 3. System Load & Average Waiting Time
    ax3.plot(df["time"], df["avg_wait"], color="#2ca02c", linewidth=2.0, label="Ready Queue Avg Wait")
    ax3.set_xlabel("Simulation Clock (Ticks)")
    ax3.set_ylabel("Avg Wait (ticks)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right")

    # Add vertical phase shift markers
    if shift_ticks:
        for idx, shift in enumerate(shift_ticks):
            for ax in (ax1, ax2, ax3):
                ax.axvline(x=shift, color="black", linestyle=":", linewidth=1.8)
                if ax == ax1:
                    ax.text(shift + 2, ax.get_ylim()[1] * 0.85, f"Phase {idx+2} Shift", color="black", fontweight="bold", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_q_table_heatmap(agent, save_path=None, ax=None):
    set_paper_style()
    q_table = agent.q_table
    visits = getattr(agent, "visits", None)
    if visits is not None:
        visited = np.asarray(visits) > 0
    else:
        visited = np.any(q_table != 0, axis=1)

    best_actions = np.argmax(q_table, axis=1)  # fine on its own, just needs masking downstream

    n = 9
    grid_q = np.full((n, n), np.nan)
    labels = np.empty((n, n), dtype=object)
    for s in range(81):
        cpu, rem = divmod(s, 27)
        qlen, rem2 = divmod(rem, 9)
        burst, io = divmod(rem2, 3)
        row, col = cpu * 3 + qlen, burst * 3 + io
        if visited[s]:
            q_val, alpha_val = agent.get_action_params(best_actions[s])
            grid_q[row, col] = q_val
            labels[row, col] = f"q={q_val}\nα={alpha_val}"
        else:
            labels[row, col] = "no data"

    fig, ax = (plt.subplots(figsize=(13, 13)) if ax is None else (ax.figure, ax))
    masked = np.ma.masked_invalid(grid_q)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    for r in range(n):
        for c in range(n):
            ax.text(c, r, labels[r, c], ha="center", va="center",
                     fontsize=8, fontweight="bold",
                     color="black" if np.isnan(grid_q[r, c]) else "white")

    ax.set_title("RL-Sched Learned Policy Mapping (State -> Action)", fontweight="bold")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"B{b} I{i}" for b in range(3) for i in range(3)], rotation=45, ha="right")
    ax.set_yticklabels([f"C{c} Q{q}" for c in range(3) for q in range(3)])

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_gantt_chart(
    timeline: List[Dict[str, Any]],
    max_ticks: int = 150,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Figure]:
    """
    Renders an execution Gantt chart showing process scheduling, context switches, and CPU idle periods.
    """
    set_paper_style()
    filtered = [e for e in timeline if e["start"] < max_ticks]
    if not filtered:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.figure
    
    # Assign colors to PIDs
    unique_pids = sorted(list({e["pid"] for e in filtered if e["pid"] is not None}))
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.get_cmap("tab20")
    pid_color = {pid: cmap(i % 20) for i, pid in enumerate(unique_pids)}

    for event in filtered:
        start = event["start"]
        duration = min(event["end"], max_ticks) - start
        if duration <= 0:
            continue

        etype = event["type"]
        pid = event["pid"]

        if etype == "CPU":
            color = pid_color.get(pid, "#4C72B0")
            label_text = f"P{pid}" if duration >= 3 else ""
            ax.barh(0, duration, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.5)
            if label_text:
                ax.text(start + duration / 2, 0, label_text, ha="center", va="center", color="white", fontweight="bold", fontsize=8)
        elif etype == "CS":
            ax.barh(0, duration, left=start, height=0.6, color="#e74c3c", hatch="//", edgecolor="black", linewidth=0.5)
        elif etype == "IDLE":
            ax.barh(0, duration, left=start, height=0.6, color="#bdc3c7", edgecolor="black", linewidth=0.5)

    ax.set_yticks([])
    ax.set_xlim(0, max_ticks)
    ax.set_xlabel("Clock Ticks")
    ax.set_title(f"Simulation CPU Execution Trace (First {max_ticks} Ticks)", fontweight="bold")
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#4C72B0", edgecolor="black", label="Process Execution"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#e74c3c", hatch="//", edgecolor="black", label="Context Switch"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#bdc3c7", edgecolor="black", label="CPU Idle")
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
