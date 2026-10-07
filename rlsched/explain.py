"""
rlsched/explain.py: interpretability helpers + one-call Streamlit panel.
Usage in app.py:  render_explainer(run_agent, history, phase_starts)
"""
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

CPU = ["Low", "Med", "High"]
QUE = ["Short", "Med", "Long"]
BUR = ["Short", "Med", "Long"]
IOR = ["Low", "Bal", "High"]


def policy_df(agent) -> pd.DataFrame:
    """One row per state: decoded features, greedy action, confidence margin, visited flag."""
    q = agent.q_table
    visits = getattr(agent, "visits", None)
    visited = (visits > 0) if visits is not None and visits.sum() > 0 else np.any(q != 0, axis=1)
    rows = []
    for s in range(agent.num_states):
        c, r = divmod(s, 27)
        ql, r = divmod(r, 9)
        b, i = divmod(r, 3)
        a = int(np.argmax(q[s]))
        srt = np.sort(q[s])
        qu, al = agent.actions[a]
        rows.append(dict(
            state=s, bc=c, bq=ql, bb=b, bi=i, a=a, quantum=qu, alpha=al,
            row=f"CPU {CPU[c]} | Queue {QUE[ql]}", col=f"Burst {BUR[b]} | I/O {IOR[i]}",
            label=f"{qu}/{al}" if visited[s] else "-",
            margin=round(float(srt[-1] - srt[-2]), 3), visited=bool(visited[s]),
        ))
    return pd.DataFrame(rows)


def policy_heatmap(df: pd.DataFrame, field: str = "quantum") -> alt.Chart:
    """9x9 grid covering all 81 states. Unvisited states are grey (their Q-values are just init)."""
    base = alt.Chart(df).encode(
        x=alt.X("col:N", title=None, sort=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("row:N", title=None, sort=None),
    )
    rect = base.mark_rect(stroke="white").encode(
        color=alt.condition("datum.visited", alt.Color(f"{field}:Q", scale=alt.Scale(scheme="blues")),
                            alt.value("#e5e7eb")),
        tooltip=["state", "quantum", "alpha", "margin", "visited"],
    )
    txt = base.mark_text(fontSize=11).encode(text="label:N")
    return (rect + txt).properties(height=330, title=f"Greedy policy: {field} (label = q/alpha, grey = never visited)")


def extract_rules(df: pd.DataFrame, depth: int = 3):
    """Distil the table into a small decision tree. Returns (rules_text, fidelity) or None."""
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        return None
    d = df[df.visited]
    if len(d) < 4:
        return None
    feats = ["bc", "bq", "bb", "bi"]
    names = ["cpu_util(0=L,2=H)", "queue_len(0=S,2=L)", "avg_burst(0=S,2=L)", "io_ratio(0=L,2=H)"]
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(d[feats], d["a"])
    return export_text(tree, feature_names=names), tree.score(d[feats], d["a"])


def phase_layer(starts):
    """Vertical dashed rules for workload phase shifts; add with `chart + phase_layer(starts)`."""
    return alt.Chart(pd.DataFrame({"t": starts})).mark_rule(strokeDash=[4, 4], color="black").encode(x="t:Q")


def render_explainer(agent, history, phase_starts=None):
    st.markdown('<div class="section-title">Interpretability: why does RL-Sched do what it does?</div>',
                unsafe_allow_html=True)
    df = policy_df(agent)
    t1, t2, t3, t4 = st.tabs(["Why this action?", "Policy map", "Extracted rules", "State coverage"])

    with t1:
        if not history:
            st.info("Run a simulation first.")
        else:
            i = st.slider("Decision #", 0, len(history) - 1, min(len(history) - 1, 5))
            h = history[i]
            lab = agent.get_state_labels(h["state"])
            st.markdown(f"**t = {h['time']}** | state **{h['state']}**: {lab['cpu']}, {lab['queue']}, "
                        f"{lab['burst']}, {lab['io']}")
            qv = h.get("q_values", agent.q_table[h["state"]].tolist())
            qdf = pd.DataFrame({
                "action": [f"q={a[0]}, a={a[1]}" for a in agent.actions],
                "Q": qv,
                "chosen": [k == int(np.argmax(qv)) for k in range(len(qv))],
            })
            st.altair_chart(alt.Chart(qdf).mark_bar().encode(
                x=alt.X("action:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
                y="Q:Q",
                color=alt.condition("datum.chosen", alt.value("#10b981"), alt.value("#94a3b8")),
                tooltip=["action", "Q"]).properties(height=260), use_container_width=True)
            top2 = np.sort(qv)[-2:]
            st.caption(f"Chosen: q={h['quantum']}, alpha={h['aging_rate']} | margin over runner-up: "
                       f"{top2[1] - top2[0]:.3f} (small margin = low confidence).")
            parts = h.get("reward_parts")
            if parts:
                st.markdown("**Reward received for the previous action, by term**")
                st.bar_chart(pd.Series(parts))

    with t2:
        which = st.radio("Show", ["quantum", "alpha"], horizontal=True)
        st.altair_chart(policy_heatmap(df, which), use_container_width=True)
        st.caption("Sanity check: you would expect smaller quanta for short-burst/high-I/O states and "
                   "larger quanta for long-burst CPU-bound states.")

    with t3:
        res = extract_rules(df)
        if res is None:
            st.info("Needs scikit-learn and at least a few visited states (`pip install scikit-learn`).")
        else:
            text, fid = res
            st.code(text)
            st.caption(f"Depth-3 tree reproduces the Q-table's greedy action on {fid:.0%} of visited states.")

    with t4:
        cnt = Counter(h["state"] for h in history)
        c1, c2 = st.columns(2)
        c1.metric("States visited in this run", f"{len(cnt)} / {agent.num_states}")
        c2.metric("States with learned (non-zero) Q", f"{int(df.visited.sum())} / {agent.num_states}")
        if cnt:
            vdf = pd.DataFrame({"state": list(cnt), "decisions": list(cnt.values())})
            st.altair_chart(alt.Chart(vdf).mark_bar(color="#2563EB").encode(
                x=alt.X("state:O", title="State index"), y="decisions:Q").properties(height=220),
                use_container_width=True)
