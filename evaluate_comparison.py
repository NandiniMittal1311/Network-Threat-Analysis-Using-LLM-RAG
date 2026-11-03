import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

INPUT_FILE = "results_comparison.csv"

def to_binary_llm(threat_type):
    """Convert the LLM's free-text threat_type into a binary attack/benign label."""
    if pd.isna(threat_type):
        return 0
    t = str(threat_type).strip().lower()
    return 0 if t in ("benign", "normal", "n/a", "none") else 1

def main():
    df = pd.read_csv(INPUT_FILE)
    df["is_attack_true"] = df["is_attack_true"].astype(int)
    df["llm_pred"] = df["llm_threat_type"].apply(to_binary_llm)

    methods = {
        "Rule-based":        "rule_pred",
        "Isolation Forest":  "iso_pred",
        "LLM + RAG":         "llm_pred",
    }

    print(f"{'Method':<20}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Accuracy':<12}")
    print("-" * 68)
    rows = []
    for name, col in methods.items():
        y_true = df["is_attack_true"]
        y_pred = df[col]
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        print(f"{name:<20}{p:<12.3f}{r:<12.3f}{f1:<12.3f}{acc:<12.3f}")
        rows.append({"method": name, "precision": p, "recall": r, "f1": f1, "accuracy": acc})

    # Also show per-attack-type breakdown for the LLM's specific classification (not just binary)
    print("\nLLM threat_type vs true label (exact match check, informational):")
    df["llm_exact_match"] = df.apply(
        lambda row: str(row["llm_threat_type"]).strip().lower() in str(row["true_label"]).strip().lower()
        or str(row["true_label"]).strip().lower() in str(row["llm_threat_type"]).strip().lower(),
        axis=1
    )
    print(f"LLM threat_type loosely matches true label in {df['llm_exact_match'].mean()*100:.1f}% of cases")

    pd.DataFrame(rows).to_csv("results_summary_metrics.csv", index=False)
    print("\nSaved results_summary_metrics.csv")

if __name__ == "__main__":
    main()
