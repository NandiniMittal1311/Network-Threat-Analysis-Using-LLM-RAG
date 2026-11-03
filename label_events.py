import csv
from datetime import datetime
from collections import defaultdict

INPUT_FILE = "parsed_events.csv"
OUTPUT_FILE = "labeled_events.csv"

FAIL_THRESHOLD = 4       # more than this many fails...
TIME_WINDOW_SEC = 10     # ...within this many seconds = brute force

def parse_ts(ts):
    return datetime.fromisoformat(ts)

def label_rows(rows):
    # Group failed attempts by source_ip, sorted by time
    fails_by_ip = defaultdict(list)
    for row in rows:
        if row["result"] == "failed":
            fails_by_ip[row["source_ip"]].append(row)

    bruteforce_pids = set()
    for ip, fails in fails_by_ip.items():
        fails.sort(key=lambda r: parse_ts(r["timestamp"]))
        for i, row in enumerate(fails):
            window = [r for r in fails if abs((parse_ts(r["timestamp"]) - parse_ts(row["timestamp"])).total_seconds()) <= TIME_WINDOW_SEC]
            if len(window) > FAIL_THRESHOLD:
                for r in window:
                    bruteforce_pids.add(r["pid"])

    for row in rows:
        row["label"] = "ssh_bruteforce" if row["pid"] in bruteforce_pids else "normal"
    return rows

if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        rows = list(csv.DictReader(f))

    rows = label_rows(rows)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} labeled rows to {OUTPUT_FILE}")
    for r in rows:
        print(r["timestamp"], r["result"], r["label"])
