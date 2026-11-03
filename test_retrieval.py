import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INDEX_FILE = "data/vector_index.faiss"
METADATA_FILE = "data/vector_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

def load_store():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["metadata"]

def search(query, model, index, chunks, metadata, top_k=TOP_K):
    q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx],
            "source": metadata[idx]["source"],
            "distance": float(dist)
        })
    return results

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    index, chunks, metadata = load_store()

    # A sample query mimicking a flagged suspicious event description
    test_query = "high packet rate flood to a single destination IP, many SYN flags no ACK"

    results = search(test_query, model, index, chunks, metadata)
    print(f"Query: {test_query}\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (source: {r['source']}, distance: {r['distance']:.4f}) ---")
        print(r["text"][:300])
        print()
