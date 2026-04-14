import json
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ✅ Use THIS file (important)
LOG_FILE = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"

FIG_DIR = PROJECT_ROOT / "phase4" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_logs(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data


def main():
    records = load_logs(LOG_FILE)

    workflows = []

    # ✅ FIXED LOOP
    for r in records:
        wf = r.get("workflow_id") or r.get("workflow") or r.get("action")
        if wf:
            workflows.append(wf)

    if len(workflows) == 0:
        print("⚠️ No workflow data found.")
        return

    counts = Counter(workflows)

    print("Workflow counts:", counts)

    plt.figure()
    plt.bar(counts.keys(), counts.values())

    plt.xlabel("Workflow")
    plt.ylabel("Count")
    plt.title("Workflow Distribution")

    plt.savefig(FIG_DIR / "workflow_distribution.png", dpi=300)
    plt.savefig(FIG_DIR / "workflow_distribution.pdf", dpi=300)
    plt.close()

    print("✅ Saved workflow_distribution.png/pdf")


if __name__ == "__main__":
    main()