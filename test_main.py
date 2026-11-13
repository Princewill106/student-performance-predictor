import random
import subprocess
import sys
import unittest

import main


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, "make_sample.py"], check=True)

    def test_recovers_known_weights(self):
        rng = random.Random(0)
        X = [[1.0, rng.random() * 10, rng.random() * 5] for _ in range(50)]
        y = [3 + 2 * x[1] - 1.5 * x[2] for x in X]
        w = main.fit(X, y)
        for got, want in zip(w, [3, 2, -1.5]):
            self.assertAlmostEqual(got, want, places=6)

    def test_model_quality(self):
        _, X, y = main.load("sample_students.csv")
        w = main.fit(X, y)
        r2, mae = main.metrics(y, main.predict(w, X))
        self.assertGreater(r2, 0.85)


if __name__ == "__main__":
    unittest.main()
