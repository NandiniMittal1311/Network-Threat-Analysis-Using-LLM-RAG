import pandas as pd

INPUT_FILE = "data/cicids/combine.csv"

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    # Column names have leading spaces — strip them
    df.columns = df.columns.str.strip()
    return df

if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns)[:10], "...")
    print("\nLabel counts:")
    print(df["Label"].value_counts())
