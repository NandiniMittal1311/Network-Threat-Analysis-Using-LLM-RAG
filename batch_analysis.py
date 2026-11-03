import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SAMPLE_FILE = "data/cicids/sample.csv"
INDEX_FILE = "data/vector_index.faiss"
METADATA_FILE = "data/vector_metadata.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 3

# How many events of EACH kind (benign + each attack type) to sample for the LLM comparison.
# LLM calls are the slow/costly step, so we keep this small but balanced across classes.
N_PER_CLASS = 5

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def load_store():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["metadata"]


def retrieve_context(event_description, index, chunks, metadata, top_k=TOP_K):
    q_vec = embed_model.encode([event_description], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_vec, top_k)
    return [chunks[idx] for idx in indices[0]]


def describe_event(row, feature_cols):
    # Flow Duration and IAT fields in CICIDS2017 are recorded in MICROSECONDS.
    # Convert to seconds and label units explicitly so the LLM doesn't misread scale.
    duration_sec = row.get("Flow Duration", 0) / 1_000_000
    iat_mean_sec = row.get("Flow IAT Mean", 0) / 1_000_000
    iat_std_sec = row.get("Flow IAT Std", 0) / 1_000_000

    parts = [
        f"Destination Port: {row.get('Destination Port')}",
        f"Flow Duration (seconds): {duration_sec:.4f}",
        f"Total Fwd Packets: {row.get('Total Fwd Packets')}",
        f"Total Backward Packets: {row.get('Total Backward Packets')}",
        f"Flow Bytes/s: {row.get('Flow Bytes/s')}",
        f"Flow Packets/s: {row.get('Flow Packets/s')}",
        f"SYN Flag Count: {row.get('SYN Flag Count')}",
        f"ACK Flag Count: {row.get('ACK Flag Count')}",
        f"FIN Flag Count: {row.get('FIN Flag Count')}",
        f"Flow IAT Mean (seconds, avg gap between packets): {iat_mean_sec:.6f}",
        f"Flow IAT Std (seconds): {iat_std_sec:.6f}",
    ]
    return "; ".join(parts)

def build_prompt(event_description, retrieved_context):
    context_block = "\n\n".join(f"[Reference {i+1}]\n{c}" for i, c in enumerate(retrieved_context))
    return f"""You are a network security analyst assistant. Analyze the flagged network flow below using
ONLY the reference material provided plus standard security knowledge. All duration/timing values below
are already converted to SECONDS — do not reinterpret their scale. Respond with STRICT JSON ONLY,
no preamble, no markdown, no backticks — just a raw JSON object with exactly these fields:

{{
  "threat_type": "<short label, e.g. DoS, PortScan, DDoS, Bot, Infiltration, Benign>",
  "severity": <integer 0-10>,
  "reasoning": "<2-3 sentences, referencing the event's specific values>",
  "mitre_technique": "<MITRE ATT&CK technique ID and name if applicable, else 'N/A'>",
  "recommended_action": "<concrete next step>"
}}

REFERENCE MATERIAL:
{context_block}

FLAGGED NETWORK FLOW:
{event_description}

Respond with ONLY the JSON object."""


def call_llm(prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
                reasoning_effort="low",
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "json_parse_failed", "raw_output": raw}
        except Exception as e:
            print(f"  >>> LLM call failed: {e}")
            if attempt < retries:
                time.sleep(2)
                continue
            return {"error": str(e)}


def load_and_prepare():
    df = pd.read_csv(SAMPLE_FILE, low_memory=False)
    df.columns = df.columns.str.strip()
    df["is_attack"] = (df["Label"] != "BENIGN").astype(int)

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != "is_attack"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = df["is_attack"].mean()
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    raw_preds = iso.predict(X_scaled)
    df["iso_pred"] = np.where(raw_preds == -1, 1, 0)  # 1 = flagged as attack by Isolation Forest

    # Simple rule baseline proxy for flow data: flag very high packet rate OR very high SYN count
    # relative to ACK (works as a stand-in rule since we don't have live traffic here)
    df["rule_pred"] = (
        (df["Flow Packets/s"] > df["Flow Packets/s"].quantile(0.95)) |
        ((df["SYN Flag Count"] > 0) & (df["ACK Flag Count"] == 0) & (df["SYN Flag Count"] >= 3))
    ).astype(int)

    return df, feature_cols


def sample_events(df, n_per_class):
    parts = []
    for label, group in df.groupby("Label"):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=42))
    return pd.concat(parts).reset_index(drop=True)


def main():
    print("Loading data, training Isolation Forest, computing rule baseline...")
    df, feature_cols = load_and_prepare()
    index, chunks, metadata = load_store()

    sampled = sample_events(df, N_PER_CLASS)
    print(f"Running LLM+RAG analysis on {len(sampled)} sampled events "
          f"(balanced across {sampled['Label'].nunique()} labels)...")

    results = []
    for i, row in sampled.iterrows():
        event_desc = describe_event(row, feature_cols)
        context = retrieve_context(event_desc, index, chunks, metadata)
        prompt = build_prompt(event_desc, context)
        llm_result = call_llm(prompt)

        results.append({
            "true_label": row["Label"],
            "is_attack_true": int(row["is_attack"]),
            "rule_pred": int(row["rule_pred"]),
            "iso_pred": int(row["iso_pred"]),
            "llm_threat_type": llm_result.get("threat_type", "N/A"),
            "llm_severity": llm_result.get("severity", None),
            "llm_reasoning": llm_result.get("reasoning", ""),
            "llm_mitre": llm_result.get("mitre_technique", ""),
            "llm_action": llm_result.get("recommended_action", ""),
        })
        print(f"[{i+1}/{len(sampled)}] true={row['Label']:20s} "
              f"rule={row['rule_pred']} iso={row['iso_pred']} "
              f"llm={llm_result.get('threat_type', 'N/A')}")
        time.sleep(0.5)  # be polite to the free-tier rate limit

    results_df = pd.DataFrame(results)
    results_df.to_csv("results_comparison.csv", index=False)
    print("\nSaved results_comparison.csv")
    print("\nSample of results:")
    print(results_df[["true_label", "rule_pred", "iso_pred", "llm_threat_type"]].to_string())


if __name__ == "__main__":
    main()
