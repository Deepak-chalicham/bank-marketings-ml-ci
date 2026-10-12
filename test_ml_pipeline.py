import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(
            os.path.exists("bank-additional-full.csv")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("bank_marketing_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_accuracy_is_valid(self):

        with open(
            "metrics.json",
            "r"
        ) as file:

            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(
            accuracy,
            0.0
        )

        self.assertLessEqual(
            accuracy,
            1.0
        )

    def test_model_prediction(self):

        model = joblib.load(
            "bank_marketing_model.pkl"
        )

        sample = pd.DataFrame([{
            "age": 56,
            "job": "housemaid",
            "marital": "married",
            "education": "basic.4y",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "telephone",
            "month": "may",
            "day_of_week": "mon",
            "duration": 261,
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.1,
            "cons.price.idx": 93.994,
            "cons.conf.idx": -36.4,
            "euribor3m": 4.857,
            "nr.employed": 5191.0
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )


if __name__ == "__main__":
    unittest.main()