"""
Run once from the project root:  python apply_patches.py
Edits q_learning.py, rl_scheduler.py, app.py and the benchmark script in place.
Originals are saved next to them as *.bak. Nothing is written unless every edit matches.
"""
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

bench = None
for f in glob.glob(os.path.join(ROOT, "benchmarks", "*.py")):
    if "shift_ticks=[300, 550]" in open(f, encoding="utf-8").read():
        bench = f

PATCHES = {
    os.path.join(ROOT, "rlsched", "q_learning.py"): ("self.visits", [
        ("        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)",
         "        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)\n"
         "        self.visits = np.zeros(self.num_states, dtype=int)", 1),
        ("        self.q_table[state, action] += lr_to_use * td_error",
         "        self.q_table[state, action] += lr_to_use * td_error\n"
         "        self.visits[state] += 1", 1),
        ('            "actions": self.actions\n',
         '            "actions": self.actions,\n'
         '            "visits": self.visits.tolist()\n', 1),
        ('        self.gamma = data.get("gamma", self.gamma)',
         '        self.gamma = data.get("gamma", self.gamma)\n'
         '        self.visits = np.array(data.get("visits", [0] * self.num_states), dtype=int)', 1),
    ]),
    os.path.join(ROOT, "rlsched", "rl_scheduler.py"): ("reward_parts", [
        ("        return float(reward)",
         "        self.last_parts = {\n"
         '            "wait": -self.w_wait * norm_wait_penalty,\n'
         '            "throughput": self.w_throughput * tp_reward,\n'
         '            "fairness": self.w_fairness * fairness,\n'
         '            "starvation": -self.w_starve * starve_penalty,\n'
         "        }\n"
         "        return float(reward)", 1),
        ('                    "avg_wait": telemetry["avg_current_wait"]\n',
         '                    "avg_wait": telemetry["avg_current_wait"],\n'
         '                    "q_values": self.agent.q_table[current_state].tolist(),\n'
         '                    "reward_parts": dict(getattr(self, "last_parts", {}))\n', 1),
    ]),
    os.path.join(ROOT, "app.py"): ("render_explainer", [
        ("import os\nimport sys\n", "import copy\nimport os\nimport sys\n", 1),
        ("from rlsched.visualizer import plot_q_table_heatmap, plot_gantt_chart",
         "from rlsched.visualizer import plot_q_table_heatmap, plot_gantt_chart\n"
         "from rlsched.explain import render_explainer", 1),
        ("    agent = QLearningAgent(seed=42)",
         "    agent = QLearningAgent(learning_rate=0.15, discount_factor=0.90, epsilon=1.0,\n"
         "                           epsilon_min=0.05, epsilon_decay=0.985, seed=42)", 1),
        ("        for ep in range(100):", "        for ep in range(250):", 1),
        ("    schedulers = [\n        FCFSScheduler(),",
         "    run_agent = copy.deepcopy(agent)  # online learning must not modify the shared agent\n"
         "    schedulers = [\n        FCFSScheduler(),", 1),
        ("RLScheduler(agent, decision_interval=decision_interval,",
         "RLScheduler(run_agent, decision_interval=decision_interval,", 1),
        ("Throughput (%/tick)", "Throughput (per 100 ticks)", 3),
        ("""    st.markdown('<div class="section-title">Q-Table Policy Heatmap (Interpretable RL)</div>', unsafe_allow_html=True)\n""",
         "", 1),
        ("    fig_heat = plot_q_table_heatmap(agent)\n"
         "    if fig_heat is not None:\n"
         "        st.pyplot(fig_heat, use_container_width=True)\n"
         "        plt.close(fig_heat)",
         "    phase_starts = None\n"
         '    if workload_type == "Dynamic Phase Shift":\n'
         "        _p1 = max(5, num_processes // 3)\n"
         "        _p2 = max(5, num_processes // 2)\n"
         "        phase_starts = [workload[_p1].arrival_time, workload[_p1 + _p2].arrival_time]\n"
         "    render_explainer(run_agent, history, phase_starts)", 1),
    ]),
}
if bench:
    PATCHES[bench] = ("dynamic_workload[20]", [
        ("shift_ticks=[300, 550]",
         "shift_ticks=[dynamic_workload[20].arrival_time, dynamic_workload[50].arrival_time]", 1),
    ])
else:
    print("Note: benchmark script with shift_ticks=[300, 550] not found; skipping that fix.")

if not os.path.exists(os.path.join(ROOT, "rlsched", "explain.py")):
    sys.exit("Put explain.py inside the rlsched folder first.")

new_sources = {}
for path, (marker, edits) in PATCHES.items():
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    src = open(path, encoding="utf-8").read()
    if marker in src:
        sys.exit(f"{os.path.basename(path)} looks already patched. Nothing changed.")
    for old, new, n in edits:
        found = src.count(old)
        if found != n:
            sys.exit(f"{os.path.basename(path)}: expected {n} match(es), found {found} for:\n{old[:90]}\nNothing changed.")
        src = src.replace(old, new)
    new_sources[path] = src

for path, src in new_sources.items():
    shutil.copy(path, path + ".bak")
    open(path, "w", encoding="utf-8").write(src)
    print("patched", os.path.relpath(path, ROOT))
print("Done. Next: pip install scikit-learn, re-run the benchmark script, then streamlit run app.py")
