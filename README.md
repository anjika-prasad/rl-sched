# RL-Sched: A Reinforcement Learning-Based Adaptive CPU Scheduler for Dynamic Workloads

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RL-Sched** is an intelligent, interpretable, data-driven operating system CPU scheduler designed for heterogeneous and shifting workloads. Unlike classical fixed heuristics (FCFS, SJF, Round Robin) or single-parameter adaptive schedulers from published literature (e.g. RLBMCS / DQRR), **RL-Sched** formulates scheduling parameter regulation as a Markov Decision Process (MDP) and employs **tabular Q-learning** to jointly adapt both the **time quantum ($q$)** and the **priority aging rate ($\alpha$)**.

---

## 🎯 Research Motivation & Identified Literature Gap

Classical CPU scheduling algorithms rely on static heuristics chosen at design time:
- **FCFS** causes convoy bottlenecks behind long jobs.
- **SJF (SRTF)** minimizes average waiting time but suffers catastrophic starvation of long jobs under high contention.
- **Round Robin (RR)** depends heavily on a static quantum: small quanta inflate context-switch overhead; large quanta degrade into FCFS convoy delays.
- **Prior Published Adaptive Approaches (e.g. RLBMCS / Shrivastava 2010)**: Only tune a single parameter (e.g. quantum alone) or use black-box Deep Q-Networks (DQN) that are difficult to inspect and verify.

### Key Contributions of RL-Sched
1. **Joint Two-Parameter Action Space**: Simultaneously regulates time quantum ($q \in \{2, 4, 8, 16\}$ ticks) and aging rate ($\alpha \in \{0.0, 0.5, 1.0, 2.0\}$), recognizing that preemption granularity and starvation mitigation interact.
2. **Composite Multi-Objective Reward**: Explicitly balances normalized waiting time minimization, throughput maximization, Jain's fairness index, and starvation penalties.
3. **Interpretability by Construction**: Employs tabular Q-learning with discretized system state ($3 \times 3 \times 3 \times 3 = 81$ states) rather than black-box neural networks, enabling exact verification and policy heatmaps.
4. **Dynamic Workload Evaluation Protocol**: Directly measures adaptivity across mid-run workload shifts (e.g., transition from CPU-bound batch compute to interactive I/O flash crowds).

---

## 📐 Mathematical Formulation (MDP)

### State Space $\mathcal{S}$ ($|\mathcal{S}| = 81$)
The scheduler inspects continuous system telemetry at discrete decision intervals and maps it into a 4-feature discrete state index:
1. **CPU Utilization**: $\text{Low} (< 40\%) $, $\text{Medium} (40\%\text{--}80\%) $, $\text{High} (> 80\%) $
2. **Ready Queue Contention**: $\text{Short} (\le 2)$, $\text{Medium} (3\text{--}6)$, $\text{Long} (> 6)$
3. **Average Burst Estimate**: $\text{Short} (< 6)$, $\text{Medium} (6\text{--}15)$, $\text{Long} (> 15)$
4. **I/O-to-CPU Ratio**: $\text{Low} (< 25\%)$, $\text{Balanced} (25\%\text{--}60\%)$, $\text{High} (> 60\%)$

$$\text{State Index} = b_{\text{cpu}} \cdot 27 + b_{\text{qlen}} \cdot 9 + b_{\text{burst}} \cdot 3 + b_{\text{io}}$$

### Action Space $\mathcal{A}$ ($|\mathcal{A}| = 16$)
Coupled discrete parameter pairs:
$$\mathcal{A} = \{ (q, \alpha) \mid q \in \{2, 4, 8, 16\}, \; \alpha \in \{0.0, 0.5, 1.0, 2.0\} \}$$

### Multi-Objective Reward Function $\mathcal{R}$
$$R = - w_1 \cdot \overline{W}_{\text{norm}} + w_2 \cdot \text{Throughput}_{\text{norm}} + w_3 \cdot \text{Jain's Fairness} - w_4 \cdot \text{Starvation Penalty}$$

### Tabular Bellman Update
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \eta \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

---

## 📊 Benchmark Results

Evaluated across identical workload traces with explicit context-switch overhead ($c=1$ tick):

### Scenario 4: Dynamic Workload Phase Shift (Adaptation Stress Test)
*Phase 1: Heavy CPU batch $\to$ Phase 2: Sudden I/O interactive flash crowd $\to$ Phase 3: Heterogeneous mixed background load*

| Scheduler | Avg Wait Time (ticks) | Avg Turnaround Time (ticks) | Jain's Fairness Index | Starvation Count | Context Switches |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FCFS** | 1061.10 | 1214.39 | 0.4948 | 0 | 0 |
| **SJF (SRTF)** | 523.16 | 709.24 | 0.8150 | **5 (High Starvation)** | 88 |
| **Round Robin ($q=4$)** | 1222.47 | 1638.50 | 0.8878 | 0 | 357 |
| **RLBMCS / Dyn-RR** *(Rinku et al.)* | 1234.91 | 1618.59 | 0.8702 | 0 | 289 |
| **RL-Sched (Proposed)** | **892.60** | **1176.17** | **0.8838** | **0 (Zero Starvation)** | 332 |

> **Key Finding**: In dynamic shifting environments, **RL-Sched reduces Average Turnaround Time by 27.3%** compared to published literature (RLBMCS) and Round Robin, while maintaining **zero starvation** (unlike SJF, which completely abandoned 5 processes).

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python -m unittest discover -s tests -v
```

### 3. Run Training and Full Benchmark Suite
```bash
python benchmarks/run_experiments.py
```
This trains the Q-learning agent, benchmarks all 5 schedulers, exports `benchmark_summary.csv`, and generates publication-grade figures in `benchmarks/results/`:
- `learned_policy_heatmap.png` (2D policy mapping)
- `dynamic_adaptation_curve.png` (Continuous parameter adaptation across workload shifts)
- `dynamic_workload_comparison.png` (4-panel comparison bar charts)
- `dynamic_gantt_chart.png` (Detailed CPU schedule timeline)

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Provides:
- Real-time scenario switcher (Dynamic shifts, CPU batch, I/O burst, Mixed).
- Interactive Gantt chart inspection.
- State-action Q-table lookup and policy heatmap.
- Side-by-side metric tables with automated best-performer highlighting.

---

## 🏗️ Architecture for Future Frontend Integration
The simulator and metrics engine are decoupled from the visualization layer. A Node.js / React frontend can interface via:
1. Standard JSON export from `sim.timeline` and `adaptation_history`.
2. REST / WebSocket endpoints using FastAPI (`uvicorn`) for live animated ready-queue execution.
