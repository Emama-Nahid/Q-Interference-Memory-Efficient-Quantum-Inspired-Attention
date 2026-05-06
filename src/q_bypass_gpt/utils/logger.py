from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "metrics.jsonl"
        self.csv_path = self.out_dir / "metrics.csv"
        self._csv_headers_written = self.csv_path.exists()

    def log(self, metrics: dict[str, Any]) -> None:
        payload = {"timestamp": time.time(), **metrics}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        fieldnames = list(payload.keys())
        write_header = not self._csv_headers_written
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._csv_headers_written = True
            writer.writerow(payload)
