import unittest

from src.config import TARGET_COLUMN
from src.data_prep import build_feature_frame, generate_demo_transactions, split_train_validation_test


class DataPrepTest(unittest.TestCase):
    def test_demo_dataset_has_expected_target_and_columns(self):
        df = generate_demo_transactions(n_samples=500, fraud_rate=0.02)

        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertIn("Amount", df.columns)
        self.assertIn("Time", df.columns)
        self.assertEqual(len(df), 500)

    def test_build_feature_frame_excludes_target(self):
        df = generate_demo_transactions(n_samples=500, fraud_rate=0.02)
        features = build_feature_frame(df)

        self.assertNotIn(TARGET_COLUMN, features.columns)
        self.assertIn("Amount", features.columns)

    def test_split_train_validation_test_keeps_rows_partitioned(self):
        df = generate_demo_transactions(n_samples=1000, fraud_rate=0.03)
        x_train, x_valid, x_test, y_train, y_valid, y_test = split_train_validation_test(df)

        total_rows = len(x_train) + len(x_valid) + len(x_test)
        self.assertEqual(total_rows, len(df))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_valid), len(y_valid))
        self.assertEqual(len(x_test), len(y_test))


if __name__ == "__main__":
    unittest.main()
