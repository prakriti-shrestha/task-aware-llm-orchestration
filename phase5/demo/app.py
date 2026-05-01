"""
Task-Aware LLM Orchestration — demo UI.

    streamlit run phase5/demo/app.py

Three tabs:
  1. Route a task      — pick from dropdown, see features + UCB + winner
  2. Compare workflows — W1/W2/W3 outputs side by side for that task
  3. Results dashboard — Pareto, regret, distribution + running stats

All data is read from phase5/demo/demo_data.json (prepare it first with
`python -m phase5.demo.prepare_demo_data`).  No live Gemini calls.
"""
from __future__ import annotations
import json
import sys
from html import escape as _html_escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Imports from project
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase3"))

DEMO_DIR = Path(__file__).resolve().parent
DEMO_JSON = DEMO_DIR / "demo_data.json"
FIGURES_DIR = DEMO_DIR.parent / "figures"

ARMS = ["W1", "W2", "W3"]
ARM_COLOR = {"W1": "#4C9F70", "W2": "#F5A623", "W3": "#D0021B"}
ARM_LABEL = {
    "W1": "W1 — single-pass (cheap)",
    "W2": "W2 — CoT + self-consistency (balanced)",
    "W3": "W3 — critique-revise (expensive)",
}

# ── Streamlit config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task-Aware LLM Orchestration",
    page_icon="⚙️",
    layout="wide",
)


# ── Data loading ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_demo_data() -> dict:
    if not DEMO_JSON.exists():
        st.error(
            f"demo_data.json not found at {DEMO_JSON}\n\n"
            f"Run this first:\n\n```\npython -m phase5.demo.prepare_demo_data\n```"
        )
        st.stop()
    with open(DEMO_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def get_task_options(_tasks: list[dict]) -> list[dict]:
    """Read curated_tasks.DEMO_TASK_IDS, else fall back to auto_select."""
    from phase5.demo.curated_tasks import DEMO_TASK_IDS, auto_select
    if DEMO_TASK_IDS:
        by_id = {t["task_id"]: t for t in _tasks}
        return [by_id[tid] for tid in DEMO_TASK_IDS if tid in by_id]
    return auto_select(_tasks, n_per_class=4)


def summarise_task(t: dict, idx: int) -> str:
    """Short dropdown label."""
    txt = t["task_text"].strip().replace("\n", " ")
    if len(txt) > 80:
        txt = txt[:77] + "…"
    return f"{idx+1:02d}. [{t['task_class']:9s}] {txt}"


# ── Header ───────────────────────────────────────────────────────────────
data = load_demo_data()
tasks_all = data["tasks"]
options = get_task_options(tasks_all)

st.title("Task-aware LLM orchestration")
st.caption(
    f"Contextual-bandit routing over {data['meta']['n_tasks']} tasks · "
    f"feature dim = {data['meta']['feature_dim']} · "
    f"α = {data['meta']['alpha']} · λ = {data['meta']['lambda']}"
)

# ── Task picker (shared across tabs) ─────────────────────────────────────
st.sidebar.header("Pick a task")
choice = st.sidebar.selectbox(
    "Select one of the curated tasks below.",
    range(len(options)),
    format_func=lambda i: summarise_task(options[i], i),
)
t = options[choice]

# Greyed-out custom text box (non-cached tasks)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Custom task input is disabled in this build — routing requires cached "
    "workflow outputs, which only exist for the 150 evaluation tasks."
)
st.sidebar.text_area("Custom task (read-only)", value="", disabled=True, height=80)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "1. Route a task",
    "2. Compare workflows",
    "3. Results dashboard",
])


# ═════════════════════════════════════════════════════════════════════════
#  TAB 1 — Route a task
# ═════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Task")
        st.markdown(
            f"**Class:** `{t['task_class']}`  ·  **ID:** `{t['task_id']}`"
        )
        st.info(t["task_text"])

        if t.get("ground_truth"):
            gt = t["ground_truth"]
            if isinstance(gt, (dict, list)):
                gt = json.dumps(gt, indent=2)[:300]
            st.caption(f"Ground truth: `{str(gt)[:300]}`")

        st.subheader("Feature vector")
        vec = np.array(t["feature_vector"])
        st.caption(
            f"{len(vec)}-dim sentence-transformer embedding "
            f"(all-MiniLM-L6-v2). Showing first 64 components."
        )
        # Show first 64 dims as a bar strip
        df_feat = pd.DataFrame({
            "dim": range(64),
            "value": vec[:64],
        })
        st.bar_chart(df_feat, x="dim", y="value", height=160)

    with col_right:
        st.subheader("LinUCB decision")
        ucb = t["ucb_at_final"]
        winner = t["linucb_choice"]
        oracle = t["oracle_choice"]

        # Big winner card
        is_correct = winner == oracle
        emoji = "✅" if is_correct else "⚠️"
        col_a, col_b = st.columns(2)
        col_a.metric("LinUCB picks", winner, help=ARM_LABEL[winner])
        col_b.metric(
            "Oracle (best retrospectively)",
            oracle,
            delta=f"{emoji} {'match' if is_correct else 'disagrees'}",
            delta_color="normal" if is_correct else "inverse",
        )

        # UCB score bars
        st.caption("UCB scores (higher = more attractive)")
        df_ucb = pd.DataFrame({
            "arm": ARMS,
            "score": [ucb[a] for a in ARMS],
        })
        st.bar_chart(df_ucb, x="arm", y="score", height=220)

        # Arm-by-arm breakdown
        st.caption("Observed reward per arm on this task (from cache):")
        rows = []
        for arm in ARMS:
            row = t["arms"][arm]
            rows.append({
                "arm": arm,
                "quality": f"{row['quality']:.3f}",
                "cost (chars)": row["cost"],
                "reward": f"{row['reward']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
#  TAB 2 — Compare workflows
# ═════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"All three workflows on this task")
    st.caption(
        f"Showing outputs for task **`{t['task_id']}`** (class: `{t['task_class']}`). "
        "Each box shows the actual output produced by that workflow during our "
        "n=150 evaluation run.  Quality is the blended score "
        "(exact match / code execution / NLI). Cost is in generated characters."
    )
    st.info(t["task_text"])

    cols = st.columns(3)
    for col, arm in zip(cols, ARMS):
        row = t["arms"][arm]
        with col:
            is_winner = arm == t["linucb_choice"]
            is_oracle = arm == t["oracle_choice"]
            badges = []
            if is_winner:
                badges.append("🎯 LinUCB pick")
            if is_oracle:
                badges.append("⭐ Oracle pick")
            badge_line = "  ·  ".join(badges) if badges else " "

            st.markdown(f"### {arm}")
            st.caption(ARM_LABEL[arm].split(" — ", 1)[1])
            st.caption(badge_line)

            m1, m2 = st.columns(2)
            m1.metric("Quality", f"{row['quality']:.3f}")
            m2.metric("Cost", f"{row['cost']:,}")
            st.metric("Reward (λ=0.5)", f"{row['reward']:.3f}")

            output = row["output_text"] or "(no output logged)"
            if len(output) > 1200:
                output = output[:1200] + "\n\n… (truncated)"
            # Render output as read-only markdown in a bordered container
            # rather than text_area — text_area keys cause stale-state issues
            # in Streamlit when switching between tasks rapidly.
            with st.container(border=True, height=260):
                st.markdown(
                    f"<div style='font-family: ui-monospace, Consolas, monospace; "
                    f"font-size: 0.85rem; white-space: pre-wrap; line-height: 1.4;'>"
                    f"{_html_escape(output)}</div>",
                    unsafe_allow_html=True,
                )


# ═════════════════════════════════════════════════════════════════════════
#  TAB 3 — Results dashboard
# ═════════════════════════════════════════════════════════════════════════
with tab3:
    # ── Per-task context ("where does this task sit?") ────────────────────
    # Tab 3 is mostly aggregate, but we show a small panel showing how the
    # currently-selected task compares to the other 149.
    st.subheader("Where this task sits")
    st.caption(
        f"Quick context for the currently-selected task **`{t['task_id']}`** "
        f"(`{t['task_class']}`) before we zoom out to the full 150-task results."
    )

    # Compute this-task stats
    chosen_arm = t["linucb_choice"]
    this_reward = t["arms"][chosen_arm]["reward"]
    this_cost = t["arms"][chosen_arm]["cost"]
    this_agrees = chosen_arm == t["oracle_choice"]

    # Compute percentile of this task's reward against all 150
    all_rewards = sorted(
        tt["arms"][tt["linucb_choice"]]["reward"] for tt in tasks_all
    )
    n = len(all_rewards)
    rank = sum(1 for r in all_rewards if r < this_reward)
    percentile = round(100 * rank / max(n - 1, 1))

    # In-class routing share for the current task's class
    class_tasks = [tt for tt in tasks_all if tt["task_class"] == t["task_class"]]
    class_w23_share = round(
        100 * sum(1 for tt in class_tasks if tt["linucb_choice"] in ("W2", "W3"))
        / max(len(class_tasks), 1)
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "This task's reward",
        f"{this_reward:.3f}",
        help="Reward LinUCB earned on the currently-selected task at λ=0.5",
    )
    p2.metric(
        "Reward percentile",
        f"{percentile}th",
        help=f"This task's reward is higher than {percentile}% of all 150 tasks "
             f"(min={all_rewards[0]:.3f}, max={all_rewards[-1]:.3f})",
    )
    p3.metric(
        "Agrees with Oracle?",
        "Yes ✅" if this_agrees else "No ⚠️",
        help=f"LinUCB picked {chosen_arm}; Oracle picked {t['oracle_choice']}",
    )
    p4.metric(
        f"{t['task_class'].title()} → W2/W3",
        f"{class_w23_share}%",
        help=f"Across all {len(class_tasks)} {t['task_class']} tasks, this is "
             f"the share routed to a heavier workflow.",
    )

    st.markdown("---")

    st.subheader("Aggregate results (n=150)")
    st.caption(
        "These four headline numbers and the figures below are computed over "
        "all 150 evaluation tasks — they do not change with the dropdown selection."
    )

    # Compute running stats across ALL tasks, not just the curated picks
    total_linucb_cost = sum(
        tt["arms"][tt["linucb_choice"]]["cost"] for tt in tasks_all
    )
    total_w3_cost = sum(tt["arms"]["W3"]["cost"] for tt in tasks_all)
    total_w1_cost = sum(tt["arms"]["W1"]["cost"] for tt in tasks_all)

    mean_linucb_q = np.mean([
        tt["arms"][tt["linucb_choice"]]["quality"] for tt in tasks_all
    ])
    mean_w3_q = np.mean([tt["arms"]["W3"]["quality"] for tt in tasks_all])
    mean_w1_q = np.mean([tt["arms"]["W1"]["quality"] for tt in tasks_all])

    cost_savings_pct = 100 * (1 - total_linucb_cost / max(total_w3_cost, 1))
    quality_retention_pct = 100 * (mean_linucb_q / max(mean_w3_q, 1e-9))

    agree = sum(
        1 for tt in tasks_all if tt["linucb_choice"] == tt["oracle_choice"]
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tasks routed", len(tasks_all))
    k2.metric("Cost vs Always-W3", f"−{cost_savings_pct:.0f}%",
              help=f"LinUCB: {total_linucb_cost:,}  ·  Always-W3: {total_w3_cost:,}")
    k3.metric("Quality vs Always-W3", f"{quality_retention_pct:.0f}%",
              help=f"LinUCB mean: {mean_linucb_q:.3f}  ·  Always-W3: {mean_w3_q:.3f}")
    k4.metric("Agrees with Oracle", f"{100*agree/len(tasks_all):.0f}%",
              help=f"{agree} of {len(tasks_all)} tasks")

    st.markdown("---")

    # Embed the three main figures
    fig_specs = [
        ("workflow_distribution.png", "Routing distribution by task class",
         "Reasoning tasks get materially more W2/W3; QA and code stay cheap."),
        ("pareto_frontier.png", "Cost–quality Pareto frontier",
         "Each purple point is LinUCB at a different λ; red/green are fixed baselines."),
        ("regret_curves.png", "Cumulative regret (3 seeds, 95% CI)",
         "LinUCB vs ε-greedy vs Random. Epsilon-greedy remains competitive at N=150."),
    ]
    for fname, title, caption in fig_specs:
        st.markdown(f"#### {title}")
        st.caption(caption)
        path = FIGURES_DIR / fname
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"Figure not found at {path}")
        st.markdown("")

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Cached demo · no live Gemini calls · "
    "`phase5/demo/app.py` · Prakriti Shrestha, Aryan Kedia, Aayusha Dhakal · 2026"
)