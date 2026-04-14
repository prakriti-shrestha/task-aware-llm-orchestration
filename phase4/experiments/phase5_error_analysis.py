import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"

def load_logs(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue  # skip bad lines safely
    return data


def main():
    records = load_logs(LOG_FILE)

    # ✅ Keep only records that have BOTH task_id and reward
    records = [r for r in records if "reward" in r and "task_id" in r]

    if len(records) == 0:
        print("⚠️ No valid records found. Check your log file.")
        return

    # ── Build oracle (best reward per task) ──
    oracle = {}
    for r in records:
        tid = r["task_id"]
        oracle[tid] = max(oracle.get(tid, -1), r["reward"])

    # ── Compute regret ──
    rows = []
    for r in records:
        tid = r["task_id"]
        regret = oracle[tid] - r["reward"]

        rows.append({
            "task": r.get("task_text", "")[:100],
            "workflow": r.get("workflow_id", "N/A"),
            "quality": r.get("quality_score", 0),
            "cost": r.get("cost_tokens", 0),
            "regret": round(regret, 4),
        })

    if len(rows) == 0:
        print("⚠️ No rows generated after processing.")
        return

    # ── Sort worst cases ──
    rows.sort(key=lambda x: x["regret"], reverse=True)

    print("\n🔴 TOP 10 WORST CASES:\n")

    for i, row in enumerate(rows[:10]):
        print(f"{i+1}. Regret={row['regret']}")
        print(f"   Workflow: {row['workflow']}")
        print(f"   Quality: {row['quality']}")
        print(f"   Cost: {row['cost']}")
        print(f"   Task: {row['task']}\n")


if __name__ == "__main__":
    main()