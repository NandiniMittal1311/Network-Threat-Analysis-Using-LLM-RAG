import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

INPUT_FILE = "data/cicids/sample.csv"

def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Binary ground truth: BENIGN = 0 (normal), anything else = 1 (attack)
    df["is_attack"] = (df["Label"] != "BENIGN").astype(int)

    # Keep only numeric feature columns, drop Label
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != "is_attack"]

    X = df[feature_cols].copy()

    # Replace inf/-inf (common in flow data, e.g. division by zero duration) with NaN, then drop or fill
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    y = df["is_attack"].values
    return X, y, df

def run_isolation_forest(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination = expected proportion of attacks, estimated from the data itself
    contamination = y.mean()
    print(f"Estimated contamination (attack proportion): {contamination:.4f}")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # IsolationForest predicts 1 = normal, -1 = anomaly. Convert to match our labels (1 = attack).
    raw_preds = model.predict(X_scaled)
    preds = np.where(raw_preds == -1, 1, 0)

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y, preds))
    print("\nClassification report:")
    print(classification_report(y, preds, target_names=["normal", "attack"]))

    return model, scaler

if __name__ == "__main__":
    X, y, df = load_and_clean(INPUT_FILE)
    print("Feature matrix shape:", X.shape)
    print("Attack rate in data:", y.mean())
    run_isolation_forest(X, y)
