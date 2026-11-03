import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

INDEX_FILE = "data/vector_index.faiss"
METADATA_FILE = "data/vector_metadata.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 3

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
    retrieved = []
    for idx in indices[0]:
        retrieved.append(chunks[idx])
    return retrieved


def describe_event(event_row):
    """Turn a flagged flow/log row into a short natural-language description for retrieval + LLM."""
    parts = []
    for key, value in event_row.items():
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


def build_prompt(event_description, retrieved_context):
    context_block = "\n\n".join(f"[Reference {i+1}]\n{c}" for i, c in enumerate(retrieved_context))
    prompt = f"""You are a network security analyst assistant. Analyze the flagged event below using ONLY the
reference material provided plus standard security knowledge. Respond with STRICT JSON ONLY, no preamble,
no markdown formatting, no backticks — just a raw JSON object with exactly these fields:

{{
  "threat_type": "<short label, e.g. DoS, PortScan, SSH Brute Force, Botnet, Infiltration, Benign>",
  "severity": <integer 0-10>,
  "reasoning": "<2-3 sentences explaining why, referencing the event's specific values>",
  "mitre_technique": "<MITRE ATT&CK technique ID and name if applicable, else 'N/A'>",
  "recommended_action": "<concrete next step, e.g. block source IP, monitor, escalate to analyst>"
}}

REFERENCE MATERIAL:
{context_block}

FLAGGED EVENT:
{event_description}

Respond with ONLY the JSON object."""
    return prompt


def analyze_event(event_row, index, chunks, metadata):
    event_description = describe_event(event_row)
    retrieved_context = retrieve_context(event_description, index, chunks, metadata)
    prompt = build_prompt(event_description, retrieved_context)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
        reasoning_effort="low",
    )

    raw_text = response.choices[0].message.content.strip()
    # Strip accidental markdown fences if the model adds them anyway
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {"error": "Failed to parse JSON", "raw_output": raw_text}

    result["_retrieved_context"] = retrieved_context
    return result


if __name__ == "__main__":
    index, chunks, metadata = load_store()

    # Test with a sample SSH brute-force-style event (matches your own earlier real data)
    test_event = {
        "source_ip": "127.0.0.1",
        "user": "victim",
        "result": "failed",
        "fail_count_in_window": 5,
        "window_seconds": 10,
        "note": "5 failed SSH logins within 10 seconds from same source IP"
    }

    result = analyze_event(test_event, index, chunks, metadata)
    print(json.dumps(result, indent=2))
