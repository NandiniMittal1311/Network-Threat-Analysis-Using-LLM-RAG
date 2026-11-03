import pandas as pd

INPUT_FILE = "data/cicids/combine.csv"
OUTPUT_FILE = "data/cicids/sample.csv"
CHUNK_SIZE = 50_000
SAMPLE_FRAC = 0.05

def build_sample(path, out_path):
    first_chunk = True
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if "Label" not in chunk.columns:
            print("WARNING: no Label column found, columns are:", list(chunk.columns))
            continue
        sampled_parts = []
        for label, group in chunk.groupby("Label"):
            sampled_parts.append(group.sample(frac=SAMPLE_FRAC, random_state=42))
        sampled = pd.concat(sampled_parts)
        sampled.to_csv(out_path, mode="w" if first_chunk else "a",
                        header=first_chunk, index=False)
        first_chunk = False
    print(f"Sample written to {out_path}")

if __name__ == "__main__":
    build_sample(INPUT_FILE, OUTPUT_FILE)
