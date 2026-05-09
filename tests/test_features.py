import unittest

import pandas as pd

from src.features import add_risk_bucket, engineer_features, feature_columns


class FeatureEngineeringTest(unittest.TestCase):
    def test_engineer_features_adds_amount_and_time_features(self):
        df = pd.DataFrame(
            {
                "Time": [0, 3600, 7200],
                "Amount": [10.0, 20.0, 30.0],
                "Class": [0, 1, 0],
            }
        )

        result = engineer_features(df)

        self.assertIn("amount_log", result.columns)
        self.assertIn("amount_zscore", result.columns)
        self.assertIn("hour", result.columns)
        self.assertIn("hour_sin", result.columns)
        self.assertIn("hour_cos", result.columns)

    def test_feature_columns_excludes_target_and_prediction_outputs(self):
        df = pd.DataFrame(
            {
                "Amount": [10.0],
                "Class": [0],
                "risk_score": [0.2],
                "prediction": [0],
            }
        )

        self.assertEqual(feature_columns(df), ["Amount"])

    def test_add_risk_bucket_creates_expected_labels(self):
        df = pd.DataFrame({"risk_score": [0.1, 0.3, 0.7, 0.95]})

        result = add_risk_bucket(df)

        self.assertEqual(result["risk_bucket"].tolist(), ["baixo", "moderado", "alto", "critico"])


if __name__ == "__main__":
    unittest.main()

