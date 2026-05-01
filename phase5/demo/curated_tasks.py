"""
Hand-picked tasks for the demo dropdown.

After running `prepare_demo_data.py`, inspect the generated demo_data.json
and pick ~12 task_ids that make the demo punchy. Good picks:
  - An easy QA where LinUCB correctly picks W1 (cheap ≈ good)
  - A hard reasoning task where LinUCB correctly picks W2 or W3
  - A code task where routing matters
  - A task where LinUCB AGREES with Oracle (success case)
  - A task where LinUCB DISAGREES with Oracle (honest limitation)

This file is imported by app.py. Edit the list below.

If you leave DEMO_TASK_IDS = None, the app will auto-select 12 tasks:
4 from each class, balanced between LinUCB-right and LinUCB-wrong.
"""

# Edit this list once you've inspected demo_data.json.
# Each string must match a task_id from phase5_all_arms.jsonl.
DEMO_TASK_IDS: list[str] | None = None


# ── Fallback auto-selector (used when DEMO_TASK_IDS is None) ─────────────
def auto_select(tasks: list[dict], n_per_class: int = 4) -> list[dict]:
    """Pick n_per_class tasks from each of qa/reasoning/code,
    balancing correct (LinUCB == Oracle) and wrong routing."""
    from collections import defaultdict
    by_class: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_class[t["task_class"]].append(t)

    picked: list[dict] = []
    for cls in ("qa", "reasoning", "code"):
        bucket = by_class.get(cls, [])
        right = [t for t in bucket if t["linucb_choice"] == t["oracle_choice"]]
        wrong = [t for t in bucket if t["linucb_choice"] != t["oracle_choice"]]
        half = n_per_class // 2
        # Deterministic: sort then slice
        right.sort(key=lambda t: t["task_id"])
        wrong.sort(key=lambda t: t["task_id"])
        picked.extend(right[:half])
        picked.extend(wrong[:n_per_class - half])
    return picked