import re
import csv
from datetime import datetime

INPUT_FILE = "ssh_raw.log"
OUTPUT_FILE = "parsed_events.csv"

# Matches lines like:
# 2026-10-12T07:06:47.966131+00:00 DESKTOP-OJN3I7D sshd[9026]: Failed password for victim from 127.0.0.1 port 40802 ssh2
LOG_PATTERN = re.compile(
    r"(?P<timestamp>[\d\-T:.+]+)\s+\S+\s+sshd\[(?P<pid>\d+)\]:\s+"
    r"(?P<result>Failed password|Accepted password)\s+for\s+"
    r"(?P<user>\S+)\s+from\s+(?P<ip>[\d.]+)\s+port\s+(?P<port>\d+)"
)

def parse_file(input_path):
    rows = []
    with open(input_path, "r") as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if not match:
                continue  # skip lines that don't match (e.g. sudo lines)
            data = match.groupdict()
            rows.append({
                "timestamp": data["timestamp"],
                "pid": data["pid"],
                "result": "accepted" if data["result"] == "Accepted password" else "failed",
                "user": data["user"],
                "source_ip": data["ip"],
                "source_port": data["port"],
            })
    return rows

def write_csv(rows, output_path):
    if not rows:
        print("No rows parsed — check your regex against the raw log format.")
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    rows = parse_file(INPUT_FILE)
    write_csv(rows, OUTPUT_FILE)
