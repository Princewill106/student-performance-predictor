"""Regenerates sample_students.csv (synthetic, reproducible)."""
import csv
import random

rng = random.Random(1)
with open("sample_students.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["attendance", "assignments", "midterm", "study_hours", "final"])
    for _ in range(120):
        att, asg = rng.uniform(50, 100), rng.uniform(30, 100)
        mid, hrs = rng.uniform(20, 95), rng.uniform(0, 12)
        final = 0.15 * att + 0.2 * asg + 0.45 * mid + 1.5 * hrs - 12 + rng.gauss(0, 4)
        w.writerow([round(att), round(asg), round(mid), round(hrs, 1), round(min(100, max(0, final)))])
