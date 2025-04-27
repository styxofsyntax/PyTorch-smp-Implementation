import csv
import os


class CSVLogger:
    def __init__(self, base_dir, fields):
        os.makedirs(base_dir, exist_ok=True)

        self.filename = os.path.join(base_dir, "metrics.csv")
        self.fields = fields

        with open(self.filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log(self, values):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(values)
