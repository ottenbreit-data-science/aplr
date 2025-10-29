import unittest
import pandas as pd
import numpy as np
from aplr import APLRClassifier, APLRRegressor


class TestAPLRPreprocessing(unittest.TestCase):

    def test_missing_value_imputation(self):
        """Tests the automatic handling of missing values."""
        X = pd.DataFrame(
            {
                "feat1": [1, 2, np.nan, 4, 5, 1, 2, 4, 5, 3],
                "feat2": [1.1, 2.2, 3.3, np.nan, 5.5, 1.1, 2.2, 3.3, 5.5, 2.75],
                "feat3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )
        y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [-1], [-1]])

        model = APLRClassifier()
        model.fit(X, y, cv_observations=cv_observations)

        # Check if missing columns were identified
        self.assertListEqual(model.na_imputed_cols_, ["feat1", "feat2"])

        # Check if median values were calculated
        self.assertAlmostEqual(model.median_values_["feat1"], 3.0)
        self.assertAlmostEqual(model.median_values_["feat2"], 2.75)
        self.assertAlmostEqual(model.median_values_["feat3"], 55.0)

        # Check transformation of training data
        X_transformed, _ = model._preprocess_X_fit(X.copy(), [], np.array([]))
        self.assertTrue("feat1_missing" in model.final_training_columns_)
        self.assertTrue("feat2_missing" in model.final_training_columns_)
        self.assertFalse("feat3_missing" in model.final_training_columns_)

        # Check prediction with missing values
        X_test = pd.DataFrame(
            {"feat1": [np.nan, 2, 3], "feat2": [1.1, np.nan, 3.3], "feat3": [1, 2, 3]}
        )
        X_transformed_pred = model._preprocess_X_predict(X_test.copy())

        # feat1 is nan, so feat1 should be median (3.0) and feat1_missing should be 1
        self.assertAlmostEqual(X_transformed_pred[0, 0], 3.0)
        self.assertAlmostEqual(X_transformed_pred[0, 3], 1.0)

        # feat2 is nan, so feat2 should be median (2.75) and feat2_missing should be 1
        self.assertAlmostEqual(X_transformed_pred[1, 1], 2.75)
        self.assertAlmostEqual(X_transformed_pred[1, 4], 1.0)

    def test_categorical_feature_handling(self):
        """Tests the automatic handling of categorical features."""
        X = pd.DataFrame(
            {
                "numeric": [1, 2, 3, 4, 1, 2, 3, 4],
                "category": ["A", "B", "A", "C", "A", "B", "A", "C"],
            }
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])

        model = APLRClassifier()
        model.fit(X, y, cv_observations=cv_observations)

        # Check if categorical features were identified
        self.assertListEqual(model.categorical_features_, ["category"])

        # Check if OHE columns were created
        self.assertIn("category_A", model.ohe_columns_)
        self.assertIn("category_B", model.ohe_columns_)
        self.assertIn("category_C", model.ohe_columns_)

        # Check prediction with unseen categories
        X_test = pd.DataFrame(
            {"numeric": [5, 6], "category": ["B", "D"]}  # 'D' is unseen
        )
        X_transformed_pred = model._preprocess_X_predict(X_test.copy())

        # Check that the shape is correct (numeric, category_A, category_B, category_C)
        self.assertEqual(X_transformed_pred.shape[1], 4)

        # Check that category_B is 1 for the first row
        self.assertEqual(X_transformed_pred[0, 2], 1)

        # Check that the unseen category 'D' results in 0 for all category columns
        self.assertEqual(X_transformed_pred[1, 1], 0)
        self.assertEqual(X_transformed_pred[1, 2], 0)
        self.assertEqual(X_transformed_pred[1, 3], 0)

    def test_regressor_missing_values(self):
        """Test APLRRegressor with missing values."""
        X = pd.DataFrame({"feat1": [1, 2, np.nan, 4, 1, 2, np.nan, 4]})
        y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])
        model = APLRRegressor()
        model.fit(X, y, cv_observations=cv_observations)

        X_test = pd.DataFrame({"feat1": [np.nan]})
        predictions = model.predict(X_test)
        self.assertEqual(predictions.shape, (1,))

    def test_classifier_missing_values(self):
        """Test APLRClassifier with missing values."""
        X = pd.DataFrame({"feat1": [1, 2, np.nan, 4, 1, 2, np.nan, 4]})
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])
        model = APLRClassifier()
        model.fit(X, y, cv_observations=cv_observations)

        X_test = pd.DataFrame({"feat1": [np.nan]})
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), 1)

    def test_missing_value_imputation_with_sample_weight(self):
        """Tests missing value imputation with sample weights."""
        X = pd.DataFrame(
            {
                "feat1": [10, 20, np.nan, 40, 50, 10, 20, np.nan, 40, 50],
                "feat2": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        sample_weight = np.array([1, 1, 1, 10, 1, 1, 1, 1, 10, 1])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [-1], [-1]])

        model = APLRClassifier()
        model.fit(X, y, sample_weight=sample_weight, cv_observations=cv_observations)

        # Check if missing columns were identified
        self.assertListEqual(model.na_imputed_cols_, ["feat1", "feat2"])

        # Check weighted median for feat1
        # Non-missing values: [10, 20, 40, 50, 10, 20, 40, 50] with weights [1, 1, 10, 1, 1, 1, 10, 1]
        # Sorted values: [10, 10, 20, 20, 40, 40, 50, 50], sorted weights: [1, 1, 1, 1, 10, 10, 1, 1]
        # Cumulative weights: [1, 2, 3, 4, 14, 24, 25, 26], total weight: 26
        # Median is at weight 13, which falls into value 40
        self.assertAlmostEqual(model.median_values_["feat1"], 40.0)

        # Check median for feat2 (all missing)
        self.assertAlmostEqual(model.median_values_["feat2"], 0)

    def test_all_missing_in_train(self):
        """Tests behavior when a column is entirely missing in training data."""
        X_train = pd.DataFrame(
            {"feat1": [1, 2, 3, 4, 5, 6, 7, 8], "feat2": [np.nan] * 8}
        )
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])

        model = APLRRegressor()
        model.fit(X_train, y_train, cv_observations=cv_observations)

        # The median for a fully NaN column will be NaN, which is then converted to 0.
        self.assertIn("feat2", model.median_values_)
        self.assertEqual(model.median_values_["feat2"], 0)

    def test_missing_in_new_data_only(self):
        """Tests imputation when a column has missing values only in new data."""
        # Training data: feat2 has no missing values
        X_train = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feat2": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        # Define CV folds to avoid random split issues with small dataset
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])
        model = APLRRegressor()
        model.fit(X_train, y_train, cv_observations=cv_observations)

        # Check that feat2 was not identified as a column with missing values during fit
        self.assertNotIn("feat2", model.na_imputed_cols_)

        # Test data: feat2 now has a missing value
        X_test = pd.DataFrame({"feat1": [1, 2], "feat2": [15, np.nan]})

        # This call should not raise an error due to the fix
        try:
            predictions = model.predict(X_test)
            self.assertEqual(predictions.shape, (2,))
        except ValueError as e:
            self.fail(f"predict() raised ValueError unexpectedly: {e}")

    def test_new_column_with_missing_values_in_predict(self):
        """Tests behavior with a new column in predict data that has missing values."""
        X_train = pd.DataFrame({"feat1": [1, 2, 3, 4, 5, 6, 7, 8]})
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])

        model = APLRRegressor()
        model.fit(X_train, y_train, cv_observations=cv_observations)

        # X_test has a new column 'feat2' which was not in X_train
        X_test = pd.DataFrame({"feat1": [1, 2], "feat2": [10, np.nan]})

        # This should raise a ValueError because the columns in X_test do not match
        # the columns seen during training.
        with self.assertRaisesRegex(
            ValueError, "Input columns for prediction do not match training columns."
        ):
            model._preprocess_X_predict(X_test)

    def test_fit_on_single_row(self):
        """Tests that fitting on less than 2 rows raises a ValueError."""
        y_single_row = np.array([0])
        error_message = "Input X must have at least 2 rows to be fitted."

        models_to_test = [APLRRegressor(), APLRClassifier()]
        inputs_to_test = [pd.DataFrame({"feat1": [1]}), np.array([[1]])]

        for model in models_to_test:
            for X_input in inputs_to_test:
                with self.subTest(
                    model=model.__class__.__name__, input_type=type(X_input).__name__
                ):
                    with self.assertRaisesRegex(ValueError, error_message):
                        model.fit(X_input, y_single_row)


if __name__ == "__main__":
    unittest.main()
