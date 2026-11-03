import csv
from datetime import datetime
from collections import defaultdict

INPUT_FILE = "parsed_events.csv"   # unlabeled, raw parsed events

FAIL_THRESHOLD = 4
TIME_WINDOW_SEC = 10

def parse_ts(ts):
    return datetime.fromisoformat(ts)

def detect(rows):
    alerts = []
    fails_by_ip = defaultdict(list)
    for row in rows:
        if row["result"] == "failed":
            fails_by_ip[row["source_ip"]].append(row)

    for ip, fails in fails_by_ip.items():
        fails.sort(key=lambda r: parse_ts(r["timestamp"]))
        for row in fails:
            window = [r for r in fails if abs((parse_ts(r["timestamp"]) - parse_ts(row["timestamp"])).total_seconds()) <= TIME_WINDOW_SEC]
            if len(window) > FAIL_THRESHOLD:
                alerts.append({
                    "source_ip": ip,
                    "user": row["user"],
                    "fail_count": len(window),
                    "window_start": window[0]["timestamp"],
                    "window_end": window[-1]["timestamp"],
                    "rule": "ssh_bruteforce: >4 failed logins within 10s",
                })
                break  # one alert per IP is enough for this simple version
    return alerts

if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        rows = list(csv.DictReader(f))

    alerts = detect(rows)

    if not alerts:
        print("No alerts raised.")
    for a in alerts:
        print("ALERT:", a)
