import os
import csv
from datetime import datetime


class TelemetryLogger:
    def __init__(self, log_dir: str = "logs", prefix: str = "episode"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{prefix}_{ts}.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = None

    def log_step(self, data: dict):
        if self._writer is None:
            # Initialize writer with keys on first write
            self._writer = csv.DictWriter(self._file, fieldnames=list(data.keys()))
            self._writer.writeheader()
        self._writer.writerow(data)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
