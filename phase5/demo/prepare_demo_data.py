"""
One-time script: extract demo-relevant data from phase5_all_arms.jsonl
into a compact JSON for the Streamlit app to load.

Run this ONCE, before launching the app.  It takes ~5 seconds.

    python -m phase5.demo.prepare_demo_data

Output:
    phase5/demo/demo_data.json   — tasks + cached outputs, 150 tasks
    phase5/demo/policy_state.npz — LinUCB state (A, b matrices)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Repo-rooted imports work from anywhere
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase3"))

from phase5.experiments._shared import feature_vector, FEATURE_DIM
from phase4.policy.bandit import LinUCBBandit

PHASE5_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = PHASE5_DIR / "demo"
ALL_ARMS = PHASE5_DIR / "data" / "runs" / "phase5_all_arms.jsonl"
DEMO_JSON = DEMO_DIR / "demo_data.json"
POLICY_NPZ = DEMO_DIR / "policy_state.npz"

LAMBDA = 0.5          # cost penalty used in the paper's main results
C_MAX = 1000.0        # cost normaliser (chars, to match reward.py)
ALPHA = 1.0           # LinUCB exploration coef
ARMS = ["W1", "W2", "W3"]


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    if not ALL_ARMS.exists():
        raise FileNotFoundError(
            f"Cannot find {ALL_ARMS}. Run phase5_main_results.py first, or "
            f"restore from phase5/backups/day2_n50/runs/."
        )

    # ── Pass 1: group JSONL rows by task_id ──────────────────────────────
    print(f"[prep] Reading {ALL_ARMS}")
    by_task: dict[str, dict] = defaultdict(lambda: {"arms": {}})
    with open(ALL_ARMS, "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            tid = r["task_id"]
            by_task[tid]["task_id"] = tid
            by_task[tid]["task_text"] = r["task_text"]
            by_task[tid]["task_class"] = r["task_class"]
            by_task[tid]["ground_truth"] = r.get("ground_truth", "")
            by_task[tid]["arms"][r["workflow_id"]] = {
                "output_text": r.get("output_text", ""),
                "quality": float(r.get("quality_score", 0.0)),
                "cost": int(r.get("cost_tokens", 0)),
                "reward": float(r.get("reward", 0.0)),
            }

    tasks = list(by_task.values())
    # Keep only tasks with all three arms logged (sanity)
    tasks = [t for t in tasks if set(t["arms"].keys()) >= set(ARMS)]
    print(f"[prep] {len(tasks)} complete tasks (all three arms logged)")

    # ── Pass 2: compute feature vectors + train LinUCB end-to-end ────────
    print(f"[prep] Encoding tasks and training LinUCB (d={FEATURE_DIM})")
    bandit = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=ALPHA, workflows=ARMS)

    # Stable ordering = same order your main_results.py used
    tasks.sort(key=lambda t: t["task_id"])

    for t in tasks:
        x = feature_vector(t["task_text"], t["task_id"])
        t["feature_vector"] = x.tolist()

        # Train LinUCB on observed all-arms rewards
        # (We update each arm's model with the reward we have for that arm,
        # since this is the offline-simulated bandit setting.)
        for arm in ARMS:
            r = t["arms"][arm]["reward"]
            bandit.update(x, arm, r)

        # Now snapshot the UCB scores at FINAL trained state for this task
        t["ucb_at_final"] = {
            arm: float(bandit._ucb_score(arm, x)) for arm in ARMS
        }
        t["linucb_choice"] = max(t["ucb_at_final"], key=t["ucb_at_final"].get)
        t["oracle_choice"] = max(
            t["arms"].keys(), key=lambda a: t["arms"][a]["reward"]
        )

    # ── Write demo_data.json ────────────────────────────────────────────
    out = {
        "meta": {
            "n_tasks": len(tasks),
            "feature_dim": FEATURE_DIM,
            "alpha": ALPHA,
            "lambda": LAMBDA,
            "c_max": C_MAX,
            "arms": ARMS,
        },
        "tasks": tasks,
    }
    with open(DEMO_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[prep] Wrote {DEMO_JSON}  ({DEMO_JSON.stat().st_size // 1024} KB)")

    # ── Save bandit state for the 'live feature' path (custom text) ─────
    # Your bandit has a built-in save() method — use it.
    bandit.save(str(POLICY_NPZ).replace(".npz", ""))  # it appends .npz itself
    print(f"[prep] Wrote {POLICY_NPZ}")

    # ── Small summary for eyeballing ────────────────────────────────────
    agree = sum(1 for t in tasks if t["linucb_choice"] == t["oracle_choice"])
    print(f"[prep] LinUCB agrees with Oracle on {agree}/{len(tasks)} tasks "
          f"({100*agree/len(tasks):.1f}%)")

    classes = defaultdict(lambda: defaultdict(int))
    for t in tasks:
        classes[t["task_class"]][t["linucb_choice"]] += 1
    print("[prep] LinUCB routing distribution by class:")
    for cls, dist in sorted(classes.items()):
        total = sum(dist.values())
        bits = [f"{arm} {dist[arm]*100/total:4.1f}%" for arm in ARMS]
        print(f"          {cls:10s}  " + "  ".join(bits))


if __name__ == "__main__":
    main()