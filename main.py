"""Student Performance Predictor: multiple linear regression from scratch."""
import argparse
import csv
import random

FEATURES = ["attendance", "assignments", "midterm", "study_hours"]
PASS_MARK = 50


def load(path, target=True):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    X = [[1.0] + [float(r[f]) for f in FEATURES] for r in rows]
    y = [float(r["final"]) for r in rows] if target else None
    return rows, X, y


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit(X, y):
    k = len(X[0])
    XtX = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    Xty = [sum(x[i] * t for x, t in zip(X, y)) for i in range(k)]
    return solve(XtX, Xty)


def predict(w, X):
    return [sum(a * b for a, b in zip(w, x)) for x in X]


def metrics(y, p):
    mean = sum(y) / len(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, p))
    ss_tot = sum((a - mean) ** 2 for a in y)
    return 1 - ss_res / ss_tot, sum(abs(a - b) for a, b in zip(y, p)) / len(y)


def main_factor(row, means):
    gaps = {f: (float(row[f]) - means[f]) / (means[f] or 1) for f in FEATURES}
    return min(gaps, key=gaps.get)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="sample_students.csv")
    ap.add_argument("--current", default="current_students.csv")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    _, X, y = load(a.train)
    idx = list(range(len(X)))
    random.Random(42).shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    w = fit([X[i] for i in tr], [y[i] for i in tr])
    r2, mae = metrics([y[i] for i in te], predict(w, [X[i] for i in te]))
    print(f"Trained on {len(tr)} students, tested on {len(te)}: R² = {r2:.3f}, MAE = {mae:.1f} marks")
    print("Coefficients:")
    for name, coef in zip(["intercept"] + FEATURES, w):
        print(f"  {name:<12} {coef:+.3f}")
    rows, Xc, _ = load(a.current, target=False)
    means = {f: sum(float(r[f]) for r in rows) / len(rows) for f in FEATURES}
    print("\nCurrent students:")
    for r, p in zip(rows, predict(w, Xc)):
        flag = f"AT RISK: weakest factor = {main_factor(r, means)}" if p < PASS_MARK else "on track"
        print(f"  {r['name']:<16} predicted {p:5.1f}  {flag}")


if __name__ == "__main__":
    main()
