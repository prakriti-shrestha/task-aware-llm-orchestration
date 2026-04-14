"""
Phase 5 — Task-Level Error Analysis (Paper Section 4.4)

Finds the 30 worst routing decisions and prints them as a clean table.
Also saves to phase5/results/error_analysis.csv.

Columns (per handbook):
  - task_text (first 100 chars)
  - predicted_features (first 5 values)
  - chosen_workflow
  - oracle_workflow
  - quality_difference
"""

import json
import csv
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE     = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
RESULTS_DIR  = PROJECT_ROOT / "phase5" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_CSV   = RESULTS_DIR / "error_analysis.csv"

TOP_N = 30


# ── Load logs ──────────────────────────────────────────────────────────────
def load_logs(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


# ── Print a table with fixed column widths ─────────────────────────────────
def print_table(rows):
    # Column definitions: (header, key, width)
    cols = [
        ("#",             "rank",         4),
        ("Class",         "task_class",   10),
        ("Chosen",        "chosen_wf",     7),
        ("Oracle",        "oracle_wf",     7),
        ("Q-Gap",         "quality_gap",   7),
        ("Features[0:5]", "features",     28),
        ("Task (first 100 chars)",  "task_snippet",  50),
    ]

    # Header
    header = "  ".join(h.ljust(w) for h, _, w in cols)
    divider = "  ".join("-" * w for _, _, w in cols)
    print("\n" + divider)
    print(header)
    print(divider)

    for row in rows:
        line = "  ".join(str(row[k]).ljust(w) for _, k, w in cols)
        print(line)

    print(divider + "\n")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    records = load_logs(LOG_FILE)
    records = [r for r in records if "reward" in r and "task_id" in r]

    if not records:
        print("No valid records found.")
        return

    # ── Build oracle per task ──────────────────────────────────────────────
    oracle_reward   = {}
    oracle_workflow = {}
    oracle_quality  = {}

    for r in records:
        tid = r["task_id"]
        if tid not in oracle_reward or r["reward"] > oracle_reward[tid]:
            oracle_reward[tid]   = r["reward"]
            oracle_workflow[tid] = r["workflow_id"]
            oracle_quality[tid]  = r["quality_score"]

    # ── Compute regret per record ──────────────────────────────────────────
    rows = []
    for r in records:
        tid    = r["task_id"]
        regret = oracle_reward[tid] - r["reward"]
        fv     = r.get("feature_vector", [])

        rows.append({
            "task_id":       tid,
            "task_class":    r.get("task_class", "?"),
            "task_snippet":  r.get("task_text", "")[:100].replace("\n", " "),
            "chosen_wf":     r.get("workflow_id", "?"),
            "oracle_wf":     oracle_workflow[tid],
            "chosen_quality":round(r.get("quality_score", 0), 4),
            "oracle_quality":round(oracle_quality[tid], 4),
            "quality_gap":   round(oracle_quality[tid] - r.get("quality_score", 0), 4),
            "regret":        round(regret, 4),
            "features":      str([round(x, 2) for x in fv[:5]]),
        })

    rows.sort(key=lambda x: x["regret"], reverse=True)
    worst = rows[:TOP_N]

    # Add rank number
    for i, row in enumerate(worst, 1):
        row["rank"] = i

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\nTOP {TOP_N} WORST ROUTING DECISIONS")
    print_table(worst)

    # ── Save CSV ───────────────────────────────────────────────────────────
    fieldnames = [
        "rank", "task_id", "task_class", "task_snippet",
        "chosen_wf", "oracle_wf",
        "chosen_quality", "oracle_quality", "quality_gap",
        "regret", "features"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(worst)

    print(f"Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()