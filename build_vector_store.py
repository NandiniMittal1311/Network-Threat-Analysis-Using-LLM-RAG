import os
import glob
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

KB_DIR = "knowledge_base"
INDEX_FILE = "data/vector_index.faiss"
METADATA_FILE = "data/vector_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_and_chunk(kb_dir):
    chunks = []
    metadata = []
    for filepath in glob.glob(os.path.join(kb_dir, "*.txt")):
        filename = os.path.basename(filepath)
        with open(filepath, "r") as f:
            text = f.read()
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(raw_chunks):
            chunks.append(chunk)
            metadata.append({"source": filename, "chunk_id": i})
    return chunks, metadata

def build_index(chunks, model_name):
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index, model

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    chunks, metadata = load_and_chunk(KB_DIR)
    print(f"Loaded {len(chunks)} chunks from {len(set(m['source'] for m in metadata))} files")

    index, model = build_index(chunks, MODEL_NAME)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)

    print(f"Saved FAISS index to {INDEX_FILE}")
    print(f"Saved metadata to {METADATA_FILE}")
