import copy
import io
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
from aplr import APLRClassifier, APLRRegressor, APLRTuner
from aplr.aplr import _dataframe_to_cpp_dataframe


class TestOptionalSklearnDependency(unittest.TestCase):
    def test_estimators_work_without_sklearn_installed(self):
        script = textwrap.dedent("""
            import builtins

            real_import = builtins.__import__

            def import_without_sklearn(name, *args, **kwargs):
                if name == "sklearn" or name.startswith("sklearn."):
                    raise ModuleNotFoundError("sklearn intentionally blocked")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = import_without_sklearn

            from aplr import APLRClassifier, APLRRegressor

            regressor = APLRRegressor()
            classifier = APLRClassifier()
            assert regressor.__sklearn_tags__() is None
            assert classifier.__sklearn_tags__() is None

            for estimator in (regressor, classifier):
                try:
                    estimator.score(None, None)
                except ImportError as error:
                    assert "scikit-learn is required" in str(error)
                else:
                    raise AssertionError("score() should require scikit-learn")
            """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class TestSmoke(unittest.TestCase):
    def test_smoke_fit_and_predict(self):
        n_rows = 1000
        X = pd.DataFrame(
            {
                "x1": np.linspace(0.0, 1.0, n_rows),
                "x2": np.linspace(1.0, 2.0, n_rows),
            }
        )

        y_reg = 1.0 + 2.0 * X["x1"] - 0.5 * X["x2"]
        regressor = APLRRegressor(m=10, random_state=0)
        regressor.fit(X, y_reg)
        reg_predictions = regressor.predict(X.iloc[:5])
        self.assertEqual(reg_predictions.shape, (5,))

        y_clf = np.where(X["x1"] > 0.5, "A", "B")
        classifier = APLRClassifier(m=10, random_state=0)
        classifier.fit(X, y_clf)
        clf_predictions = classifier.predict(X.iloc[:5])
        self.assertEqual(len(clf_predictions), 5)


class TestVerbosityAndObjectCopy(unittest.TestCase):
    def test_regressor_verbosity_output_and_object_copy(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {
                "x1": np.linspace(0.0, 1.0, 80),
                "x2": np.linspace(1.0, 2.0, 80),
            }
        )
        y = 2.0 + 3.0 * X["x1"] - 1.5 * X["x2"] + rng.normal(0.0, 0.05, size=len(X))
        cv_observations = np.array(
            [[1] if i < 60 else [-1] for i in range(len(X))], dtype=int
        )

        quiet_model = APLRRegressor(m=10, verbosity=0)
        quiet_output = io.StringIO()
        with redirect_stdout(quiet_output):
            quiet_model.fit(X, y, cv_observations=cv_observations)
        self.assertEqual(quiet_output.getvalue(), "")

        verbose_model = APLRRegressor(m=10, verbosity=1)
        verbose_output = io.StringIO()
        with redirect_stdout(verbose_output):
            verbose_model.fit(X, y, cv_observations=cv_observations)
        self.assertTrue(verbose_output.getvalue())

        copied = copy.deepcopy(verbose_model)
        self.assertIsNot(copied, verbose_model)
        self.assertEqual(copied.verbosity, 1)
        self.assertEqual(copied.APLRRegressor.verbosity, 1)
        self.assertEqual(len(copied.predict(X.iloc[:5])), 5)

    def test_classifier_verbosity_output_and_object_copy(self):
        X = pd.DataFrame(
            {
                "x1": np.linspace(0.0, 1.0, 80),
                "x2": np.linspace(1.0, 2.0, 80),
            }
        )
        y = np.where(X["x1"] > 0.5, "A", "B")
        cv_observations = np.array(
            [[1] if i < 60 else [-1] for i in range(len(X))], dtype=int
        )

        quiet_model = APLRClassifier(m=10, verbosity=0)
        quiet_output = io.StringIO()
        with redirect_stdout(quiet_output):
            quiet_model.fit(X, y, cv_observations=cv_observations)
        self.assertEqual(quiet_output.getvalue(), "")

        verbose_model = APLRClassifier(m=10, verbosity=1)
        verbose_output = io.StringIO()
        with redirect_stdout(verbose_output):
            verbose_model.fit(X, y, cv_observations=cv_observations)
        self.assertTrue(verbose_output.getvalue())

        copied = copy.copy(verbose_model)
        self.assertIsNot(copied, verbose_model)
        self.assertEqual(copied.verbosity, 1)
        self.assertEqual(copied.APLRClassifier.verbosity, 1)
        self.assertEqual(len(copied.predict(X.iloc[:5])), 5)

    def test_unfitted_model_copy_keeps_empty_callback_state(self):
        regressor = APLRRegressor(m=10, verbosity=1)
        classifier = APLRClassifier(m=10, verbosity=1)

        regressor_copy = copy.deepcopy(regressor)
        classifier_copy = copy.copy(classifier)

        self.assertIsNot(regressor_copy, regressor)
        self.assertIsNot(classifier_copy, classifier)
        self.assertEqual(regressor_copy.verbosity, 1)
        self.assertEqual(classifier_copy.verbosity, 1)
        self.assertFalse(regressor_copy.APLRRegressor is None)
        self.assertFalse(classifier_copy.APLRClassifier is None)

        # Unfitted models should still be copyable, and a missing callback should remain a safe default.
        self.assertEqual(regressor_copy.APLRRegressor.verbosity, 1)
        self.assertEqual(classifier_copy.APLRClassifier.verbosity, 1)


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

        X_original = X.copy()
        model = APLRClassifier(preprocess=True)
        model.fit(X, y, cv_observations=cv_observations)

        # Verify that the original input DataFrame is not modified during fit
        pd.testing.assert_frame_equal(X, X_original)

        preprocessor = model.APLRClassifier.preprocessor

        # Check if missing columns were identified
        self.assertIn("feat1", preprocessor.numeric_cols_)
        self.assertIn("feat2", preprocessor.numeric_cols_)

        # Check if median values were calculated
        self.assertAlmostEqual(preprocessor.numeric_imputers_["feat1"].median_, 3.0)
        self.assertAlmostEqual(preprocessor.numeric_imputers_["feat2"].median_, 2.75)

        # Check transformation of training data
        final_cols = preprocessor.get_transformed_column_names()
        self.assertIn("feat1_is_missing", final_cols)
        self.assertIn("feat2_is_missing", final_cols)
        self.assertNotIn("feat3_is_missing", final_cols)

        # Check prediction with missing values
        X_test = pd.DataFrame(
            {"feat1": [np.nan, 2, 3], "feat2": [1.1, np.nan, 3.3], "feat3": [1, 2, 3]}
        )

        # The predict method should handle preprocessing internally.
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), 3)

        # Get column indexes dynamically to make the test robust to column order changes.
        # To verify the transformation, we can call the preprocessor's transform method directly
        X_transformed_pred, _ = preprocessor.transform(
            _dataframe_to_cpp_dataframe(X_test)
        )
        feat1_idx = final_cols.index("feat1")
        feat2_idx = final_cols.index("feat2")
        feat1_missing_idx = final_cols.index("feat1_is_missing")
        feat2_missing_idx = final_cols.index("feat2_is_missing")

        # For the first row, feat1 is nan, so it should be imputed with the median (3.0) and feat1_is_missing should be 1.
        self.assertAlmostEqual(X_transformed_pred[0, feat1_idx], 3.0)
        self.assertAlmostEqual(X_transformed_pred[0, feat1_missing_idx], 1.0)

        # For the second row, feat2 is nan, so it should be imputed with the median (2.75) and feat2_is_missing should be 1.
        self.assertAlmostEqual(X_transformed_pred[1, feat2_idx], 2.75)
        self.assertAlmostEqual(X_transformed_pred[1, feat2_missing_idx], 1.0)

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

        X_original = X.copy()
        model = APLRClassifier(preprocess=True)
        model.fit(X, y, cv_observations=cv_observations)

        # Verify that the original input DataFrame is not modified during fit
        pd.testing.assert_frame_equal(X, X_original)

        preprocessor = model.APLRClassifier.preprocessor

        # Check if categorical features were identified
        self.assertIn("category", preprocessor.categorical_cols_)

        # Check if OHE columns were created
        final_cols = preprocessor.get_transformed_column_names()
        self.assertIn("category_A", final_cols)
        self.assertIn("category_B", final_cols)
        self.assertIn("category_C", final_cols)

        # Check prediction with unseen categories
        X_test = pd.DataFrame(
            {"numeric": [5, 6], "category": ["B", "D"]}  # 'D' is unseen
        )
        X_test_original = X_test.copy()
        X_transformed_pred, _ = preprocessor.transform(
            _dataframe_to_cpp_dataframe(X_test)
        )

        # Verify that the original input DataFrame is not modified by the transform method.
        pd.testing.assert_frame_equal(X_test, X_test_original)

        # Get column indexes dynamically
        cat_A_idx = final_cols.index("category_A")
        cat_B_idx = final_cols.index("category_B")
        cat_C_idx = final_cols.index("category_C")

        # The transformed data should have columns for numeric, category_A, category_B, and category_C.
        self.assertEqual(X_transformed_pred.shape[1], 4)

        # The first test row has category 'B', so its one-hot encoded column should be 1.
        self.assertEqual(X_transformed_pred[0, cat_B_idx], 1)

        # The second test row has an unseen category 'D', so all one-hot encoded columns should be 0.
        self.assertEqual(X_transformed_pred[1, cat_A_idx], 0)
        self.assertEqual(X_transformed_pred[1, cat_B_idx], 0)
        self.assertEqual(X_transformed_pred[1, cat_C_idx], 0)

    def test_regressor_missing_values(self):
        """Test APLRRegressor with missing values."""
        X = pd.DataFrame({"feat1": [1, 2, np.nan, 4, 1, 2, np.nan, 4]})
        y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])
        model = APLRRegressor(preprocess=True)
        model.fit(X, y, cv_observations=cv_observations)

        X_test = pd.DataFrame({"feat1": [np.nan]})
        predictions = model.predict(X_test)
        self.assertEqual(predictions.shape, (1,))

    def test_classifier_missing_values(self):
        """Test APLRClassifier with missing values."""
        X = pd.DataFrame({"feat1": [1, 2, np.nan, 4, 1, 2, np.nan, 4]})
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])
        model = APLRClassifier(preprocess=True)
        model.fit(X, y, cv_observations=cv_observations)

        X_test = pd.DataFrame({"feat1": [np.nan]})
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), 1)

    def test_classifier_response_types(self):
        """Tests APLRClassifier with different response variable types."""
        X = pd.DataFrame({"feat1": np.arange(100)})

        # 1. List of strings
        y_list_str = ["A" if i % 2 == 0 else "B" for i in range(100)]
        model = APLRClassifier(random_state=0, m=10)
        model.fit(X, y_list_str)
        self.assertTrue(all(isinstance(c, str) for c in model.get_categories()))

        # 2. List of integers
        y_list_int = [i % 2 for i in range(100)]
        model = APLRClassifier(random_state=0, m=10)
        model.fit(X, y_list_int)
        self.assertTrue(all(isinstance(c, str) for c in model.get_categories()))
        self.assertIn("0", model.get_categories())

        # 3. List of mixed types
        y_list_mixed = ["A" if i % 2 == 0 else 1 for i in range(100)]
        model = APLRClassifier(random_state=0, m=10)
        model.fit(X, y_list_mixed)
        self.assertTrue(all(isinstance(c, str) for c in model.get_categories()))
        self.assertIn("1", model.get_categories())

        # 4. Numpy array of integers
        y_np_int = np.array([i % 2 for i in range(100)])
        model = APLRClassifier(random_state=0, m=10)
        model.fit(X, y_np_int)
        self.assertTrue(all(isinstance(c, str) for c in model.get_categories()))

        # 5. Numpy array of strings
        y_np_str = np.array(["A" if i % 2 == 0 else "B" for i in range(100)])
        model = APLRClassifier(random_state=0, m=10)
        model.fit(X, y_np_str)
        self.assertTrue(all(isinstance(c, str) for c in model.get_categories()))

        # 6. Pandas Series of integers
        y_pd_int = pd.Series([i % 2 for i in range(100)])
        model = APLRClassifier(random_state=0, m=10)
        model.fit(X, y_pd_int)
        self.assertTrue(all(isinstance(c, str) for c in model.get_categories()))

        # 7. Pandas Series of strings
        y_pd_str = pd.Series(["A" if i % 2 == 0 else "B" for i in range(100)])
        model = APLRClassifier(random_state=0, m=10)
        model.fit(X, y_pd_str)
        self.assertTrue(all(isinstance(c, str) for c in model.get_categories()))

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

        model = APLRClassifier(preprocess=True)
        model.fit(X, y, sample_weight=sample_weight, cv_observations=cv_observations)

        preprocessor = model.APLRClassifier.preprocessor

        # Check if columns with NaNs were correctly identified for imputation.
        # The imputer for feat1 should have seen NaNs.
        self.assertTrue(preprocessor.numeric_imputers_["feat1"].had_nans_in_fit_)
        # The imputer for feat2 should also have seen NaNs.
        self.assertTrue(preprocessor.numeric_imputers_["feat2"].had_nans_in_fit_)

        # Verify the weighted median for feat1.
        # Non-missing values: [10, 20, 40, 50, 10, 20, 40, 50] with weights [1, 1, 10, 1, 1, 1, 10, 1]
        # Sorted values: [10, 10, 20, 20, 40, 40, 50, 50], sorted weights: [1, 1, 1, 1, 10, 10, 1, 1]
        # Cumulative weights: [1, 2, 3, 4, 14, 24, 25, 26], total weight: 26
        # The median is at weight 13, which corresponds to the value 40.
        self.assertAlmostEqual(preprocessor.numeric_imputers_["feat1"].median_, 40.0)

        # Check median for feat2 (all missing). The C++ imputer defaults to 0.
        self.assertAlmostEqual(preprocessor.numeric_imputers_["feat2"].median_, 0.0)

    def test_all_missing_in_train(self):
        """Tests behavior when a column is entirely missing in training data."""
        X_train = pd.DataFrame(
            {"feat1": [1, 2, 3, 4, 5, 6, 7, 8], "feat2": [np.nan] * 8}
        )
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])

        # A column with all NaNs should be imputed with 0 by the preprocessor.
        model = APLRRegressor(preprocess=True)
        model.fit(X_train, y_train, cv_observations=cv_observations)
        preprocessor = model.APLRRegressor.preprocessor
        self.assertAlmostEqual(preprocessor.numeric_imputers_["feat2"].median_, 0.0)

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
        model = APLRRegressor(preprocess=True)
        model.fit(X_train, y_train, cv_observations=cv_observations)

        preprocessor = model.APLRRegressor.preprocessor
        # Check that feat2 was not identified as a column with missing values during fit
        self.assertFalse(preprocessor.numeric_imputers_["feat2"].had_nans_in_fit_)

        # Test data where feat2 now has a missing value.
        X_test = pd.DataFrame({"feat1": [1, 2], "feat2": [15, np.nan]})

        # This call should not raise an error due to the fix
        try:
            predictions = model.predict(X_test)
            self.assertEqual(predictions.shape, (2,))
        except ValueError as e:
            self.fail(f"predict() raised ValueError unexpectedly: {e}")

    def test_new_column_in_predict_with_preprocess_false(self):
        """
        Tests that an error is raised for a new column in predict data
        when preprocess=False.
        """
        X_train = pd.DataFrame({"feat1": [1, 2, 3, 4, 5, 6, 7, 8]})
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])

        model = APLRRegressor(preprocess=False)
        model.fit(X_train, y_train, cv_observations=cv_observations)

        # X_test has a new column 'feat2' which was not in X_train
        X_test = pd.DataFrame({"feat1": [1, 2], "feat2": [10, np.nan]})

        # With preprocess=False, the C++ backend should raise an error because the
        # number of columns in the test data (2) does not match the training data (1).
        with self.assertRaisesRegex(
            RuntimeError, "X must have 1 columns but 2 were provided."
        ):
            model.predict(X_test)

    def test_new_column_in_predict_with_preprocess_true(self):
        """
        Tests that a new column in predict data is ignored when preprocess=True.
        """
        X_train = pd.DataFrame({"feat1": [1, 2, 3, 4, 5, 6, 7, 8]})
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        cv_observations = np.array([[1], [1], [1], [1], [1], [1], [-1], [-1]])

        model = APLRRegressor(preprocess=True)
        model.fit(X_train, y_train, cv_observations=cv_observations)

        # X_test has a new column 'feat2' which was not in X_train
        X_test = pd.DataFrame({"feat1": [1, 2], "feat2": [10, np.nan]})

        # With preprocess=True, the preprocessor should ignore the extra column in the test data.
        predictions = model.predict(X_test)
        self.assertEqual(predictions.shape, (2,))

    def test_fit_on_single_row(self):
        """Tests that fitting on less than 2 rows raises a ValueError."""
        y_single_row = np.array([0])
        error_message_regressor = "X and y cannot have less than two rows."
        error_message_classifier = "The number of categories must be at least 2."

        models_to_test = [
            (APLRRegressor(), error_message_regressor),
            (APLRClassifier(), error_message_classifier),
        ]
        inputs_to_test = [pd.DataFrame({"feat1": [1]}), np.array([[1]])]

        for model, error_message in models_to_test:
            for X_input in inputs_to_test:
                with self.subTest(
                    model=model.__class__.__name__, input_type=type(X_input).__name__
                ):
                    with self.assertRaisesRegex(RuntimeError, error_message):
                        model.fit(X_input, y_single_row)

    def test_input_type_flexibility(self):
        """Tests various combinations of DataFrame/NumPy array inputs for fit and predict."""
        n_train = 100
        n_predict = 10
        n_features = 2
        np.random.seed(0)
        # Training data
        X_np_train = np.random.rand(n_train, n_features)
        # Add some missing values to the training data
        X_np_train[5, 0] = np.nan
        X_np_train[15, 1] = np.nan
        X_df_train = pd.DataFrame(
            X_np_train, columns=[f"feat{i+1}" for i in range(n_features)]
        )
        y_train = np.random.rand(n_train)
        # Prediction data
        X_np_predict = np.random.rand(n_predict, n_features)
        # Add some missing values to the prediction data
        X_np_predict[2, 1] = np.nan
        X_np_predict[7, 0] = np.nan
        X_df_predict = pd.DataFrame(
            X_np_predict, columns=[f"feat{i+1}" for i in range(n_features)]
        )
        test_cases = [
            ("df_fit_np_predict", X_df_train, X_np_predict),
            ("np_fit_df_predict", X_np_train, X_df_predict),
            ("df_fit_df_predict", X_df_train, X_df_predict),
            ("np_fit_np_predict", X_np_train, X_np_predict),
        ]
        reference_predictions = None
        for name, X_fit_input, X_predict_input in test_cases:
            with self.subTest(msg=f"Scenario: {name}"):
                fit_kwargs = {}
                if isinstance(X_fit_input, np.ndarray):
                    fit_kwargs["X_names"] = [f"feat{i+1}" for i in range(n_features)]
                model = APLRRegressor(m=10, v=0.1, random_state=0, preprocess=True)
                model.fit(X_fit_input, y_train, **fit_kwargs)
                predictions = model.predict(X_predict_input)
                # Check shape and type
                self.assertIsInstance(predictions, np.ndarray)
                self.assertEqual(predictions.shape, (n_predict,))
                if reference_predictions is None:
                    reference_predictions = predictions
                else:
                    np.testing.assert_allclose(
                        predictions,
                        reference_predictions,
                        err_msg=f"Predictions for {name} do not match reference.",
                    )

    def test_fit_df_predict_mixed_types(self):
        """
        Tests fitting on a DataFrame with mixed types and predicting on both
        DataFrame and NumPy array inputs.
        """
        np.random.seed(0)
        n_train = 100
        n_predict = 10

        # 1. Create training data with numerical, categorical, and missing values
        num_feat_nan_train = np.random.rand(n_train)
        num_feat_nan_train[np.random.choice(n_train, 10, replace=False)] = np.nan
        X_train_df = pd.DataFrame(
            {
                "num_feat_nan": num_feat_nan_train,
                "cat_feat": np.random.choice(["A", "B", "C"], n_train),
                "num_feat": np.random.rand(n_train) * 100,
            }
        )
        y_train = np.random.rand(n_train)

        # 2. Fit the model
        model = APLRRegressor(m=10, v=0.1, random_state=0, preprocess=True)
        model.fit(X_train_df, y_train)

        # 3. Create prediction data
        num_feat_nan_predict = np.random.rand(n_predict)
        num_feat_nan_predict[np.random.choice(n_predict, 2, replace=False)] = np.nan
        X_predict_df = pd.DataFrame(
            {
                "num_feat_nan": num_feat_nan_predict,
                "cat_feat": np.random.choice(
                    ["B", "C", "D"], n_predict
                ),  # 'D' is unseen
                "num_feat": np.random.rand(n_predict) * 100,
            }
        )

        # 4. Scenario 1: Predict with DataFrame
        with self.subTest(msg="Scenario: predict_with_dataframe"):
            predictions_df = model.predict(X_predict_df)
            self.assertEqual(predictions_df.shape, (n_predict,))

        # 5. Scenario 2: Predict with CppDataFrame
        with self.subTest(msg="Scenario: predict_with_numpy"):
            # Convert the pandas DataFrame to a CppDataFrame to test the C++ backend directly.
            # The C++ `predict` method handles preprocessing internally using the fitted preprocessor.
            X_predict_cpp_df = _dataframe_to_cpp_dataframe(X_predict_df)

            # Call the C++ backend's predict method, which uses the fitted C++ preprocessor.
            predictions_np = model.APLRRegressor.predict(X_predict_cpp_df)
            self.assertEqual(predictions_np.shape, (n_predict,))

        # 6. Assert that predictions are identical
        np.testing.assert_allclose(
            predictions_df, predictions_np, err_msg="Predictions do not match."
        )

    def test_unfitted_preprocessor_behavior(self):
        """
        Tests model behavior when preprocessing is disabled (preprocess=False).
        """
        # 1. Setup data
        np.random.seed(42)
        X_train_np = np.random.rand(100, 2)
        y_train = np.random.rand(100)
        X_train_df_numeric = pd.DataFrame(X_train_np, columns=["num1", "num2"])
        X_train_df_mixed = pd.DataFrame(
            {"num": [1.0], "cat": ["a"]},
        )
        y_single = np.array([1.0])

        # 2. Create a model with preprocessing disabled
        model = APLRRegressor(preprocess=False, random_state=0)

        # 3. Test fitting with a NumPy array (should work)
        try:
            model.fit(X_train_np, y_train)
            predictions = model.predict(X_train_np)
            self.assertEqual(predictions.shape, (100,))
        except RuntimeError as e:
            self.fail(f"Fitting with NumPy array and preprocess=False failed: {e}")

        # 4. Test fitting with a purely numeric DataFrame (should work)
        try:
            model = APLRRegressor(preprocess=False, random_state=0)
            model.fit(X_train_df_numeric, y_train)
            predictions = model.predict(X_train_df_numeric)
            self.assertEqual(predictions.shape, (100,))
        except Exception as e:
            self.fail(
                f"Fitting with numeric DataFrame and preprocess=False failed: {e}"
            )


class TestAPLRPythonAPI(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack(
            [np.linspace(-1.0, 1.0, 40), np.tile([-1.0, 0.0, 1.0, 2.0], 10)]
        )
        self.columns = ["signal", "level"]
        self.y = 2.0 + 1.5 * self.X[:, 0] - 0.75 * self.X[:, 1]
        self.weights = np.linspace(0.5, 1.5, len(self.y))
        self.X_df = pd.DataFrame(self.X, columns=self.columns)
        self.labels = np.where(self.y > np.median(self.y), "high", "low")

    def _fit_regressor(self, y=None, fit_kwargs=None, **kwargs):
        options = {
            "m": 12,
            "v": 0.2,
            "cv_folds": 2,
            "bins": 6,
            "random_state": 3,
            "n_jobs": 1,
            "preprocess": False,
        }
        options.update(kwargs)
        fit_options = dict(fit_kwargs or {})
        sample_weight = fit_options.pop("sample_weight", self.weights)
        return APLRRegressor(**options).fit(
            self.X,
            self.y if y is None else y,
            sample_weight=sample_weight,
            X_names=self.columns,
            **fit_options,
        )

    def _fit_classifier(self, fit_kwargs=None, **kwargs):
        options = {
            "m": 12,
            "v": 0.2,
            "cv_folds": 2,
            "bins": 6,
            "random_state": 3,
            "n_jobs": 1,
            "preprocess": False,
        }
        options.update(kwargs)
        return APLRClassifier(**options).fit(
            self.X,
            self.labels,
            sample_weight=self.weights,
            X_names=self.columns,
            **(fit_kwargs or {}),
        )

    def test_regressor_public_outputs_and_shapes(self):
        model = self._fit_regressor()
        predictions = model.predict(
            self.X[:8], cap_predictions_to_minmax_in_training=False
        )
        terms = model.calculate_terms(self.X[:8])
        local_features = model.calculate_local_feature_contribution(self.X[:8])
        local_terms = model.calculate_local_term_contribution(self.X[:8])
        selected = model.calculate_local_contribution_from_selected_terms(
            self.X[:8], [0]
        )
        self.assertEqual(predictions.shape, (8,))
        self.assertEqual(terms.shape[0], 8)
        self.assertEqual(local_features.shape, (8, 2))
        self.assertEqual(local_terms.shape, terms.shape)
        self.assertEqual(selected.shape, (8,))
        self.assertEqual(model.calculate_feature_importance(self.X[:8]).shape, (2,))
        self.assertEqual(
            model.calculate_term_importance(self.X[:8]).shape, (terms.shape[1],)
        )
        self.assertEqual(
            len(model.get_term_names()), len(model.get_term_coefficients())
        )
        self.assertEqual(len(model.get_term_affiliations()), terms.shape[1])
        self.assertEqual(len(model.get_term_main_predictor_indexes()), terms.shape[1])
        self.assertEqual(len(model.get_term_interaction_levels()), terms.shape[1])
        self.assertEqual(
            len(model.get_unique_term_affiliations()),
            len(model.get_base_predictors_in_each_unique_term_affiliation()),
        )
        self.assertTrue(np.isfinite(model.get_intercept()))
        self.assertGreaterEqual(model.get_optimal_m(), 0)
        self.assertEqual(model.get_validation_tuning_metric(), "default")
        self.assertTrue(np.isfinite(model.get_cv_error()))
        self.assertEqual(model.get_num_cv_folds(), 2)
        self.assertGreater(len(model.get_validation_error_steps()), 0)
        shape = model.get_main_effect_shape(0)
        self.assertIsInstance(shape, dict)
        model.set_intercept(model.get_intercept() + 1.0)
        shifted = model.predict(self.X[:8], cap_predictions_to_minmax_in_training=False)
        np.testing.assert_allclose(shifted - predictions, 1.0)

    def test_regressor_dataframe_and_numpy_outputs_match(self):
        model = self._fit_regressor()
        matrix_outputs = (
            model.predict(self.X[:8]),
            model.calculate_feature_importance(self.X[:8]),
            model.calculate_term_importance(self.X[:8]),
            model.calculate_local_feature_contribution(self.X[:8]),
            model.calculate_local_term_contribution(self.X[:8]),
            model.calculate_terms(self.X[:8]),
        )
        dataframe_outputs = (
            model.predict(self.X_df.iloc[:8]),
            model.calculate_feature_importance(self.X_df.iloc[:8]),
            model.calculate_term_importance(self.X_df.iloc[:8]),
            model.calculate_local_feature_contribution(self.X_df.iloc[:8]),
            model.calculate_local_term_contribution(self.X_df.iloc[:8]),
            model.calculate_terms(self.X_df.iloc[:8]),
        )
        for matrix_output, dataframe_output in zip(matrix_outputs, dataframe_outputs):
            np.testing.assert_allclose(matrix_output, dataframe_output)

    def test_regressor_parameter_round_trip_for_every_exposed_parameter(self):
        model = APLRRegressor()
        values = {
            "m": 17,
            "v": 0.13,
            "random_state": 9,
            "loss_function": "mae",
            "link_function": "identity",
            "n_jobs": 2,
            "cv_folds": 3,
            "bins": 11,
            "max_interaction_level": 2,
            "max_interactions": 8,
            "min_observations_in_split": 0.2,
            "ineligible_boosting_steps_added": 4,
            "max_eligible_terms": 5,
            "verbosity": 1,
            "dispersion_parameter": 1.7,
            "validation_tuning_metric": "mae",
            "quantile": 0.7,
            "boosting_steps_before_interactions_are_allowed": 2,
            "monotonic_constraints_ignore_interactions": True,
            "group_mse_by_prediction_bins": 4,
            "group_mse_cycle_min_obs_in_bin": 3,
            "early_stopping_rounds": 6,
            "num_first_steps_with_linear_effects_only": 2,
            "penalty_for_non_linearity": 0.1,
            "penalty_for_interactions": 0.2,
            "max_terms": 7,
            "ridge_penalty": 0.3,
            "mean_bias_correction": True,
            "faster_convergence": True,
            "preprocess": False,
            "validation_ratio": 0.25,
        }
        self.assertIs(model.set_params(**values), model)
        parameters = model.get_params()
        for name, value in values.items():
            self.assertEqual(parameters[name], value, name)
            self.assertEqual(getattr(model.APLRRegressor, name), value, name)

    def test_remaining_regression_losses_and_links(self):
        positive_y = self.y - self.y.min() + 1.0
        for loss in [
            "poisson",
            "gamma",
            "tweedie",
            "negative_binomial",
            "weibull",
            "exponential_power",
            "quantile",
        ]:
            with self.subTest(loss=loss):
                model = self._fit_regressor(
                    y=positive_y,
                    loss_function=loss,
                    dispersion_parameter=1.5,
                    quantile=0.75,
                )
                self.assertEqual(model.predict(self.X[:5]).shape, (5,))
        log_model = self._fit_regressor(y=positive_y, link_function="log")
        self.assertTrue(np.all(log_model.predict(self.X[:5]) > 0))

    def test_regressor_serial_and_parallel_python_models_match(self):
        serial = self._fit_regressor(n_jobs=1)
        parallel = self._fit_regressor(n_jobs=2)
        np.testing.assert_allclose(
            serial.predict(self.X[:10]), parallel.predict(self.X[:10])
        )
        np.testing.assert_allclose(
            serial.get_feature_importance(), parallel.get_feature_importance()
        )
        self.assertAlmostEqual(serial.get_cv_error(), parallel.get_cv_error())

    def test_classifier_public_outputs_and_parameter_round_trip(self):
        model = self._fit_classifier()
        probabilities = model.predict_class_probabilities(self.X[:8])
        predictions = model.predict(self.X[:8])
        self.assertEqual(probabilities.shape, (8, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(len(predictions), 8)
        self.assertEqual(
            model.calculate_local_feature_contribution(self.X[:8]).shape, (8, 2)
        )
        self.assertEqual(model.get_feature_importance().shape, (2,))
        self.assertEqual(model.get_validation_error_steps().ndim, 2)
        self.assertTrue(np.isfinite(model.get_cv_error()))
        self.assertEqual(
            model.get_logit_model(model.get_categories()[0]).get_num_cv_folds(), 2
        )
        self.assertEqual(model.get_params()["n_jobs"], 1)
        classifier_values = {
            "m": 15,
            "v": 0.15,
            "random_state": 4,
            "n_jobs": 2,
            "cv_folds": 3,
            "bins": 9,
            "verbosity": 1,
            "max_interaction_level": 2,
            "max_interactions": 9,
            "min_observations_in_split": 0.25,
            "ineligible_boosting_steps_added": 3,
            "max_eligible_terms": 4,
            "boosting_steps_before_interactions_are_allowed": 1,
            "monotonic_constraints_ignore_interactions": True,
            "early_stopping_rounds": 5,
            "num_first_steps_with_linear_effects_only": 1,
            "penalty_for_non_linearity": 0.1,
            "penalty_for_interactions": 0.2,
            "max_terms": 6,
            "ridge_penalty": 0.2,
            "preprocess": False,
            "validation_ratio": 0.25,
        }
        model.set_params(**classifier_values)
        for name, value in classifier_values.items():
            self.assertEqual(model.get_params()[name], value, name)
            self.assertEqual(getattr(model.APLRClassifier, name), value, name)

    def test_classifier_dataframe_and_parallel_paths(self):
        serial = self._fit_classifier(n_jobs=1)
        parallel = self._fit_classifier(n_jobs=2)
        np.testing.assert_allclose(
            serial.predict_class_probabilities(self.X_df.iloc[:8]),
            parallel.predict_class_probabilities(self.X_df.iloc[:8]),
        )
        self.assertEqual(
            serial.predict(self.X_df.iloc[:8]), parallel.predict(self.X_df.iloc[:8])
        )

    def test_python_wrapper_rejects_invalid_inputs(self):
        model = APLRRegressor(preprocess=False)
        with self.assertRaises(RuntimeError):
            model.fit(self.X, self.y[:-1])
        with self.assertRaises(RuntimeError):
            model.fit(self.X, self.y, sample_weight=-np.ones(len(self.y)))
        with self.assertRaises(RuntimeError):
            model.fit(self.X, self.y, X_names=["only_one_name"])
        with self.assertRaises(RuntimeError):
            APLRRegressor(loss_function="unknown").fit(self.X, self.y)
        with self.assertRaises(RuntimeError):
            APLRClassifier().fit(self.X, np.array(["only"] * len(self.y)))

        # 5. Test fitting with a DataFrame containing non-numeric data (should fail)
        X_train_df_mixed = self.X_df.copy()
        X_train_df_mixed["category"] = np.where(self.X_df["level"] > 0, "A", "B")
        y_single = self.y.copy()
        model = APLRRegressor(preprocess=False, random_state=0)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot convert DataFrame to matrix if it contains non-numeric columns. "
            "Please ensure all columns are numeric or set preprocess=True.",
        ):
            model.fit(X_train_df_mixed, y_single)

    def test_regressor_advanced_fit_arguments_and_custom_functions(self):
        callback_calls = []

        def custom_loss(y, predictions, sample_weight, group, other_data):
            callback_calls.append("loss")
            return np.mean((y - predictions) ** 2)

        def custom_validation(y, predictions, sample_weight, group, other_data):
            callback_calls.append("validation")
            return np.mean((y - predictions) ** 2)

        def custom_gradient(y, predictions, group, other_data):
            callback_calls.append("gradient")
            return y - predictions

        def custom_hessian(y, predictions, group, other_data):
            callback_calls.append("hessian")
            return np.ones_like(y)

        def custom_transform(linear_predictor):
            return np.exp(linear_predictor)

        model = self._fit_regressor(
            loss_function="custom_function",
            validation_tuning_metric="custom_function",
            calculate_custom_loss_function=custom_loss,
            calculate_custom_validation_error_function=custom_validation,
            calculate_custom_negative_gradient_function=custom_gradient,
            calculate_custom_hessian_function=custom_hessian,
            link_function="custom_function",
            calculate_custom_transform_linear_predictor_to_predictions_function=custom_transform,
            calculate_custom_differentiate_predictions_wrt_linear_predictor_function=custom_transform,
            calculate_custom_differentiate2_predictions_wrt_linear_predictor_function=custom_transform,
        )
        self.assertTrue(callback_calls)
        self.assertIn("loss", callback_calls)
        self.assertIn("gradient", callback_calls)
        self.assertIn("hessian", callback_calls)
        self.assertIn("validation", callback_calls)
        self.assertTrue(np.all(np.isfinite(model.predict(self.X[:5]))))
        model.remove_provided_custom_functions()
        self.assertIsNone(model.calculate_custom_loss_function)
        self.assertIsNone(model.calculate_custom_validation_error_function)
        self.assertIsNone(model.calculate_custom_negative_gradient_function)
        self.assertIsNone(model.calculate_custom_hessian_function)

    def test_regressor_advanced_fit_arguments_are_forwarded(self):
        cv_observations = np.ones((len(self.y), 2), dtype=int)
        cv_observations[:20, 0] = -1
        cv_observations[20:, 1] = -1
        other_data = np.column_stack([self.X[:, 0] ** 2])
        model = self._fit_regressor(
            max_interaction_level=2,
            max_interactions=8,
            penalty_for_non_linearity=0.1,
            penalty_for_interactions=0.1,
            ridge_penalty=0.2,
            boosting_steps_before_interactions_are_allowed=1,
            monotonic_constraints_ignore_interactions=True,
            fit_kwargs={
                "group": np.arange(len(self.y)) % 2,
                "interaction_constraints": [[0, 1]],
                "other_data": other_data,
                "predictor_learning_rates": [0.2, 0.3],
                "predictor_penalties_for_non_linearity": [0.01, 0.02],
                "predictor_penalties_for_interactions": [0.03, 0.04],
                "predictor_min_observations_in_split": [2, 2],
                "cv_observations": cv_observations,
                "prioritized_predictors_indexes": [0],
                "monotonic_constraints": [1, -1],
            },
        )
        self.assertEqual(model.get_num_cv_folds(), 2)
        self.assertEqual(model.get_cv_validation_indexes(0).size, 20)
        self.assertEqual(model.predict(self.X[:5]).shape, (5,))

    def test_regressor_shapes_names_and_affiliation_plot(self):
        model = self._fit_regressor(max_interaction_level=1)
        original_names = model.get_term_names()
        model.set_term_names(["renamed_signal", "renamed_level"])
        renamed_names = model.get_term_names()
        self.assertEqual(len(renamed_names), len(original_names))
        self.assertEqual(renamed_names[0], "Intercept")
        self.assertTrue(any("renamed_signal" in name for name in renamed_names[1:]))
        affiliations = model.get_unique_term_affiliations()
        self.assertTrue(affiliations)
        shape = model.get_unique_term_affiliation_shape(affiliations[0])
        self.assertEqual(shape.ndim, 2)
        self.assertGreater(shape.shape[0], 0)
        try:
            missing_shape = model.get_unique_term_affiliation_shape(
                "missing affiliation"
            )
        except RuntimeError:
            missing_shape = np.empty((0, 0))
        self.assertEqual(missing_shape.shape, (0, 0))
        try:
            missing_main_effect = model.get_main_effect_shape(999)
        except RuntimeError:
            missing_main_effect = {}
        self.assertEqual(missing_main_effect, {})

        try:
            import matplotlib
        except ImportError:
            self.skipTest("matplotlib is not installed")
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "shape.png")
            model.plot_affiliation_shape(
                affiliations[0], plot=False, save=True, path=output_path
            )
            self.assertTrue(os.path.exists(output_path))
        try:
            model.plot_affiliation_shape("missing affiliation", plot=False)
        except ValueError:
            pass

    def test_regressor_pickle_and_selected_term_overloads(self):
        model = self._fit_regressor()
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(
            model.predict(self.X[:8]), restored.predict(self.X[:8])
        )
        for name in (
            "m",
            "v",
            "random_state",
            "loss_function",
            "link_function",
            "cv_folds",
            "preprocess",
        ):
            self.assertEqual(restored.get_params()[name], model.get_params()[name])
        empty_selection = model.calculate_local_contribution_from_selected_terms(
            self.X[:8], []
        )
        frame_selection = model.calculate_local_contribution_from_selected_terms(
            self.X_df.iloc[:8], [0]
        )
        self.assertEqual(empty_selection.shape, (8,))
        self.assertEqual(frame_selection.shape, (8,))

    def test_classifier_multiclass_callbacks_and_edge_cases(self):
        multiclass_labels = np.array(
            [f"class_{index % 3}" for index in range(len(self.y))]
        )
        model = APLRClassifier(
            m=12, v=0.2, cv_folds=2, bins=6, random_state=3, n_jobs=1, preprocess=False
        )
        model.fit(
            self.X, multiclass_labels, sample_weight=self.weights, X_names=self.columns
        )
        probabilities = model.predict_class_probabilities(self.X[:8])
        self.assertEqual(probabilities.shape, (8, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(len(model.get_categories()), 3)
        self.assertEqual(
            len(model.get_base_predictors_in_each_unique_term_affiliation()),
            len(model.get_unique_term_affiliations()),
        )
        with self.assertRaises(RuntimeError):
            model.get_logit_model("missing category")

        with self.assertRaises(RuntimeError):
            APLRClassifier(preprocess=False).fit(
                self.X, np.array(["only"] * len(self.y))
            )
        with self.assertRaises(RuntimeError):
            APLRClassifier(preprocess=False).fit(
                self.X, self.labels, sample_weight=np.zeros(len(self.y))
            )

        callback_model = self._fit_classifier(verbosity=1)
        self.assertEqual(
            len(
                callback_model.predict(
                    self.X[:5], cap_predictions_to_minmax_in_training=True
                )
            ),
            5,
        )
        callback_model.clear_cv_results()
        self.assertEqual(
            callback_model.get_logit_model(
                callback_model.get_categories()[0]
            ).get_num_cv_folds(),
            0,
        )

    def test_python_wrapper_input_and_parameter_validation(self):
        with self.assertRaises(RuntimeError):
            self._fit_regressor(cv_folds=1)
        with self.assertRaises(RuntimeError):
            self._fit_regressor(validation_ratio=0.0)
        with self.assertRaises(RuntimeError):
            self._fit_regressor(m=0)
        with self.assertRaises((RuntimeError, TypeError)):
            self._fit_regressor(n_jobs=-1)
        with self.assertRaises(RuntimeError):
            self._fit_regressor(fit_kwargs={"interaction_constraints": [[0, 99]]})
        with self.assertRaises(RuntimeError):
            self._fit_regressor(fit_kwargs={"monotonic_constraints": [1, 1, 1]})
        with self.assertRaises(RuntimeError):
            self._fit_regressor(fit_kwargs={"sample_weight": np.ones(len(self.y) - 1)})

        with self.assertRaises(RuntimeError):
            self._fit_classifier(fit_kwargs={"interaction_constraints": [[0, 99]]})

    def test_unfitted_public_api_guards(self):
        regressor = APLRRegressor(preprocess=False)
        classifier = APLRClassifier(preprocess=False)
        with self.assertRaises(RuntimeError):
            regressor.predict(self.X)
        self.assertEqual(regressor.get_term_names(), [])
        try:
            classifier_predictions = classifier.predict(self.X)
        except RuntimeError:
            classifier_predictions = []
        self.assertEqual(len(classifier_predictions), 0)
        try:
            classifier_probabilities = classifier.predict_class_probabilities(self.X)
        except RuntimeError:
            classifier_probabilities = np.empty((0, 0))
        self.assertEqual(classifier_probabilities.shape, (0, 0))
        self.assertIsInstance(regressor.get_params(), dict)
        self.assertIsInstance(classifier.get_params(), dict)

    def test_shape_sampling_and_selected_term_validation(self):
        model = self._fit_regressor(max_interaction_level=1)
        affiliation = model.get_unique_term_affiliations()[0]
        sampled_shape = model.get_unique_term_affiliation_shape(
            affiliation, max_rows_before_sampling=1, additional_points=3
        )
        self.assertEqual(sampled_shape.ndim, 2)
        self.assertGreater(sampled_shape.shape[0], 0)
        invalid_selection = model.calculate_local_contribution_from_selected_terms(
            self.X[:5], [999]
        )
        self.assertEqual(invalid_selection.shape, (5,))
        duplicate_selection = model.calculate_local_contribution_from_selected_terms(
            self.X[:5], [0, 0]
        )
        self.assertEqual(duplicate_selection.shape, (5,))

    def test_classifier_capped_predictions_and_all_cv_cleanup(self):
        model = self._fit_classifier()
        uncapped = model.predict_class_probabilities(
            self.X[:8], cap_predictions_to_minmax_in_training=False
        )
        capped = model.predict_class_probabilities(
            self.X[:8], cap_predictions_to_minmax_in_training=True
        )
        self.assertEqual(capped.shape, uncapped.shape)
        self.assertTrue(np.all(np.isfinite(capped)))
        for category in model.get_categories():
            self.assertEqual(
                model.get_logit_model(category).get_num_cv_folds(), model.cv_folds
            )
        model.clear_cv_results()
        for category in model.get_categories():
            logit_model = model.get_logit_model(category)
            self.assertEqual(logit_model.get_num_cv_folds(), 0)
            with self.assertRaises(RuntimeError):
                logit_model.get_cv_y(0)

    def test_custom_callback_errors_and_link_execution(self):
        def raising_transform(values):
            raise ValueError("transform failure")

        with self.assertRaises(RuntimeError):
            self._fit_regressor(
                link_function="custom_function",
                calculate_custom_transform_linear_predictor_to_predictions_function=raising_transform,
            )

        def custom_transform(values):
            return np.exp(values)

        custom_model = self._fit_regressor(
            link_function="custom_function",
            calculate_custom_transform_linear_predictor_to_predictions_function=custom_transform,
            calculate_custom_differentiate_predictions_wrt_linear_predictor_function=custom_transform,
            calculate_custom_differentiate2_predictions_wrt_linear_predictor_function=custom_transform,
        )
        self.assertTrue(np.all(np.isfinite(custom_model.predict(self.X[:3]))))
        custom_model.remove_provided_custom_functions()
        self.assertIsNotNone(
            custom_model.calculate_custom_transform_linear_predictor_to_predictions_function
        )
        self.assertIsNotNone(
            custom_model.calculate_custom_differentiate_predictions_wrt_linear_predictor_function
        )
        self.assertIsNotNone(
            custom_model.calculate_custom_differentiate2_predictions_wrt_linear_predictor_function
        )

    def test_classifier_pickle_and_legacy_state_defaults(self):
        model = self._fit_classifier()
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(
            model.predict_class_probabilities(self.X[:8]),
            restored.predict_class_probabilities(self.X[:8]),
        )
        state = model.__dict__.copy()
        for field in ("ridge_penalty", "preprocess", "validation_ratio"):
            state.pop(field, None)
        restored_state = APLRClassifier.__new__(APLRClassifier)
        restored_state.__setstate__(state)
        self.assertEqual(restored_state.ridge_penalty, 0.0)
        self.assertFalse(restored_state.preprocess)
        self.assertTrue(np.isnan(restored_state.validation_ratio))

    def test_tuner_forwards_fit_kwargs(self):
        cv_observations = np.ones((len(self.y), 2), dtype=int)
        cv_observations[:20, 0] = -1
        cv_observations[20:, 1] = -1
        tuner = APLRTuner(parameters={"m": [6], "v": [0.1]}, is_regressor=True)
        tuner.fit(
            self.X,
            self.y,
            sample_weight=self.weights,
            X_names=self.columns,
            cv_observations=cv_observations,
        )
        self.assertEqual(len(tuner.get_cv_results()), 1)
        self.assertEqual(tuner.get_best_estimator().get_num_cv_folds(), 2)


class TestAPLRTunerValidation(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)})
        self.y = 1.0 + self.X["x"]

    def test_tuner_requires_fit_before_prediction(self):
        tuner = APLRTuner(parameters={"m": [5]}, is_regressor=True)
        with self.assertRaises(AttributeError):
            tuner.predict(self.X)
        with self.assertRaises(AttributeError):
            tuner.get_best_estimator()

    def test_tuner_parameter_grid_and_unknown_parameter(self):
        tuner = APLRTuner(parameters={"m": [5, 6], "v": [0.1, 0.2]})
        self.assertEqual(len(tuner.parameter_grid), 4)
        with self.assertRaises((RuntimeError, TypeError, AttributeError)):
            APLRTuner(parameters={"not_a_parameter": [1]}).fit(self.X, self.y)


class TestAPLRTuner(unittest.TestCase):
    def setUp(self):
        """Set up common data for tuner tests."""
        np.random.seed(42)
        # Regression data
        self.X_reg = pd.DataFrame(
            {"feat1": np.random.rand(50), "feat2": np.random.rand(50) * 10}
        )
        self.y_reg = self.X_reg["feat1"] + 2 * self.X_reg["feat2"] + np.random.rand(50)

        # Classification data
        self.X_clf = pd.DataFrame(
            {"feat1": np.random.rand(50), "feat2": np.random.rand(50)}
        )
        self.y_clf = (self.X_clf["feat1"] + self.X_clf["feat2"] > 1).astype(int)

    def test_tuner_for_regressor(self):
        """Tests APLRTuner with APLRRegressor."""
        parameters = {
            "max_interaction_level": [0, 1],
            "v": [0.1, 0.5],
            "loss_function": ["mse", "mae"],
            "m": [100],  # Fixed parameter for all runs
            "random_state": [0],  # Fixed parameter for all runs
        }
        tuner = APLRTuner(parameters=parameters, is_regressor=True)

        tuner.fit(self.X_reg, self.y_reg)

        # 1. Test get_best_estimator
        best_model = tuner.get_best_estimator()
        self.assertIsInstance(best_model, APLRRegressor)
        self.assertEqual(best_model.get_params()["random_state"], 0)
        self.assertEqual(best_model.get_params()["m"], 100)

        # 2. Test get_cv_results
        cv_results = tuner.get_cv_results()
        self.assertIsInstance(cv_results, list)
        self.assertEqual(len(cv_results), 8)  # 2*2*2 combinations
        self.assertIn("cv_error", cv_results[0])
        # Check if results are sorted by cv_error
        errors = [res["cv_error"] for res in cv_results]
        self.assertEqual(errors, sorted(errors))
        # Check that fixed params are in results
        self.assertEqual(cv_results[0]["m"], 100)
        # Check that string parameter was tuned
        self.assertTrue(any(r["loss_function"] == "mae" for r in cv_results))

        # Check that the best model corresponds to the lowest CV error
        self.assertEqual(best_model.get_cv_error(), cv_results[0]["cv_error"])

        # 3. Test predict
        predictions = tuner.predict(self.X_reg)
        self.assertEqual(predictions.shape, (50,))

        # 4. Test predict_class_probabilities raises error
        with self.assertRaisesRegex(
            TypeError, "predict_class_probabilities is only possible"
        ):
            tuner.predict_class_probabilities(self.X_reg)

        # 5. Test with numpy array
        tuner_np = APLRTuner(parameters=parameters, is_regressor=True)
        tuner_np.fit(self.X_reg.values, self.y_reg, X_names=["feat1", "feat2"])
        predictions_np = tuner_np.predict(self.X_reg.values)
        np.testing.assert_allclose(predictions, predictions_np)

    def test_sequential_tuning_regressor(self):
        """Tests sequential tuning for APLRRegressor."""
        parameters = {
            "v": [0.1, 0.5, 0.9],
            "max_interaction_level": [0, 1],
            "loss_function": ["mse", "mae"],
            "m": [50],
            "random_state": [0],
        }

        # Expected runs:
        # 1. Tune 'v' (3 values): 3 runs
        # 2. Tune 'max_interaction_level' (2 values): 1 new run
        # 3. Tune 'loss_function' (2 values): 1 new run
        # Total unique runs = 3 + 1 + 1 = 5.

        tuner = APLRTuner(
            parameters=parameters, is_regressor=True, sequential_tuning=True
        )
        tuner.fit(self.X_reg, self.y_reg)

        cv_results = tuner.get_cv_results()

        self.assertEqual(len(cv_results), 5)

        best_model = tuner.get_best_estimator()
        lowest_error = min(r["cv_error"] for r in cv_results)
        self.assertEqual(best_model.get_cv_error(), lowest_error)

        errors = [res["cv_error"] for res in cv_results]
        self.assertEqual(errors, sorted(errors))

        with self.assertRaisesRegex(
            ValueError, "sequential_tuning=True requires parameters to be a dictionary"
        ):
            tuner_list = APLRTuner(parameters=[{"v": [0.1]}], sequential_tuning=True)
            tuner_list.fit(self.X_reg, self.y_reg)

    def test_sequential_tuning_classifier(self):
        """Tests sequential tuning for APLRClassifier."""
        parameters = {
            "v": [0.1, 0.5, 0.9],
            "max_interaction_level": [0, 1],
            "m": [50],
            "random_state": [0],
        }

        # Expected runs:
        # 1. Tune 'v' (3 values): 3 runs
        # 2. Tune 'max_interaction_level' (2 values): 1 new run
        # Total unique runs = 3 + 1 = 4.

        tuner = APLRTuner(
            parameters=parameters, is_regressor=False, sequential_tuning=True
        )
        tuner.fit(self.X_clf, self.y_clf)

        cv_results = tuner.get_cv_results()

        self.assertEqual(len(cv_results), 4)

        best_model = tuner.get_best_estimator()
        lowest_error = min(r["cv_error"] for r in cv_results)
        self.assertEqual(best_model.get_cv_error(), lowest_error)

        errors = [res["cv_error"] for res in cv_results]
        self.assertEqual(errors, sorted(errors))

        # Check that the best model is a classifier
        self.assertIsInstance(best_model, APLRClassifier)

        # Check predict and predict_proba
        predictions = tuner.predict(self.X_clf)
        self.assertEqual(len(predictions), 50)
        self.assertTrue(all(isinstance(p, str) for p in predictions))

        probs = tuner.predict_proba(self.X_clf)
        self.assertEqual(probs.shape, (50, 2))

    def test_tuner_for_classifier(self):
        """Tests APLRTuner with APLRClassifier."""
        parameters = {
            "max_interaction_level": [0, 1],
            "v": [0.1, 0.5],
            "m": [100],
            "random_state": [0],
        }
        tuner = APLRTuner(parameters=parameters, is_regressor=False)

        tuner.fit(self.X_clf, self.y_clf)

        # 1. Test get_best_estimator
        best_model = tuner.get_best_estimator()
        self.assertIsInstance(best_model, APLRClassifier)

        # Check that the best model corresponds to the lowest CV error
        cv_results = tuner.get_cv_results()
        self.assertEqual(best_model.get_cv_error(), cv_results[0]["cv_error"])

        # 2. Test predict
        predictions = tuner.predict(self.X_clf)
        self.assertEqual(len(predictions), 50)
        self.assertTrue(all(isinstance(p, str) for p in predictions))

        # 3. Test predict_class_probabilities and predict_proba
        probs = tuner.predict_class_probabilities(self.X_clf)
        self.assertEqual(probs.shape, (50, 2))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)

        probs2 = tuner.predict_proba(self.X_clf)
        np.testing.assert_array_equal(probs, probs2)


class TestAPLRCvResults(unittest.TestCase):
    def test_cv_results_retrieval(self):
        """Replicates the C++ test for CV results retrieval and calculation."""
        # 1. Setup data
        np.random.seed(0)
        X_np = 2 * np.random.rand(100, 2) - 1
        X = pd.DataFrame(X_np, columns=[f"feat{i+1}" for i in range(X_np.shape[1])])
        y = X_np[:, 0] + X_np[:, 1] * 2 + (2 * np.random.rand(100) - 1)
        sample_weight = 1.0 + np.random.rand(100)

        cv_folds = 4
        model = APLRRegressor(m=10, v=0.1, cv_folds=cv_folds, random_state=0)

        # 2. Test that accessing data before fitting raises an error
        with self.assertRaises(RuntimeError):
            model.get_cv_y(0)

        # 3. Fit model
        model.fit(X, y, sample_weight=sample_weight)

        # 4. Test get_num_cv_folds
        self.assertEqual(model.get_num_cv_folds(), cv_folds)

        # 5. Test data retrieval and manually calculate cv_error
        sample_weight_normalized = sample_weight / sample_weight.mean()
        total_validation_obs = 0
        total_training_weight = 0.0
        fold_validation_errors_test1 = []
        fold_validation_errors_test2 = []
        fold_training_weight_sums = []

        for i in range(cv_folds):
            cv_y = model.get_cv_y(i)
            cv_preds = model.get_cv_validation_predictions(i)
            cv_weights = model.get_cv_sample_weight(i)
            cv_indexes = model.get_cv_validation_indexes(i)

            self.assertGreater(len(cv_y), 0)
            self.assertEqual(len(cv_y), len(cv_preds))
            self.assertEqual(len(cv_y), len(cv_weights))
            self.assertEqual(len(cv_y), len(cv_indexes))

            total_validation_obs += len(cv_y)

            # Test 1: Manually calculate validation error for this fold from get_cv_* methods
            validation_errors1 = (cv_y - cv_preds) ** 2
            fold_validation_error1 = np.sum(validation_errors1 * cv_weights) / np.sum(
                cv_weights
            )
            fold_validation_errors_test1.append(fold_validation_error1)

            # Test 2: Manually calculate validation error using original y/weights and returned indexes
            cv_y_from_indexes = y[cv_indexes]
            cv_weights_from_indexes = sample_weight_normalized[cv_indexes]
            validation_errors2 = (cv_y_from_indexes - cv_preds) ** 2
            fold_validation_error2 = np.sum(
                validation_errors2 * cv_weights_from_indexes
            ) / np.sum(cv_weights_from_indexes)
            fold_validation_errors_test2.append(fold_validation_error2)

            # Replicate internal logic for training weight sum
            is_validation = np.zeros(len(y), dtype=bool)
            is_validation[cv_indexes] = True
            train_weights_for_fold = sample_weight_normalized[~is_validation]
            training_weight_sum = np.sum(train_weights_for_fold)

            fold_training_weight_sums.append(training_weight_sum)
            total_training_weight += training_weight_sum

        self.assertEqual(total_validation_obs, len(y))

        # Finalize and assert for the manual cv_error calculation
        manual_cv_error1 = 0.0
        manual_cv_error2 = 0.0
        for i in range(cv_folds):
            manual_cv_error1 += fold_validation_errors_test1[i] * (
                fold_training_weight_sums[i] / total_training_weight
            )
            manual_cv_error2 += fold_validation_errors_test2[i] * (
                fold_training_weight_sums[i] / total_training_weight
            )

        self.assertAlmostEqual(manual_cv_error1, model.get_cv_error())
        self.assertAlmostEqual(manual_cv_error2, model.get_cv_error())

        # 6. Test clear_cv_results
        model.clear_cv_results()
        self.assertEqual(model.get_num_cv_folds(), 0)

        # 7. Test that accessing data after clearing raises an error
        with self.assertRaises(RuntimeError):
            model.get_cv_y(0)

        # 8. Test APLRClassifier
        y_class = np.where(y > np.mean(y), "A", "B")
        classifier = APLRClassifier(m=10, v=0.1, cv_folds=cv_folds, random_state=0)
        classifier.fit(X, y_class)

        # Check that data exists in one of the logit models
        logit_model_before_clear = classifier.get_logit_model("A")
        self.assertEqual(logit_model_before_clear.get_num_cv_folds(), cv_folds)
        self.assertGreater(len(logit_model_before_clear.get_cv_y(0)), 0)

        # Clear results and check again
        classifier.clear_cv_results()
        logit_model_after_clear = classifier.get_logit_model("A")
        self.assertEqual(logit_model_after_clear.get_num_cv_folds(), 0)

        with self.assertRaises(RuntimeError):
            logit_model_after_clear.get_cv_y(0)

    def test_cv_results_with_cv_observations(self):
        """Tests CV results when cv_observations is provided."""
        # 1. Setup data
        np.random.seed(0)
        X_np = 2 * np.random.rand(100, 2) - 1
        X = pd.DataFrame(X_np, columns=[f"feat{i+1}" for i in range(X_np.shape[1])])
        y = X_np[:, 0] + X_np[:, 1] * 2 + (2 * np.random.rand(100) - 1)

        # Create a 2-fold cv_observations matrix
        # Each column represents a fold. -1=validation, 1=training.
        cv_observations = np.ones((100, 2), dtype=int)
        # Fold 0: first 50 obs are validation, rest are training
        cv_observations[:50, 0] = -1
        # Fold 1: last 50 obs are validation, rest are training
        cv_observations[50:, 1] = -1

        model = APLRRegressor(m=10, v=0.1, random_state=0)

        # 2. Fit model with cv_observations
        model.fit(X, y, cv_observations=cv_observations)

        # 3. Check number of folds
        self.assertEqual(model.get_num_cv_folds(), 2)

        # 4. Check validation indexes
        fold0_indexes = model.get_cv_validation_indexes(0)
        fold1_indexes = model.get_cv_validation_indexes(1)

        self.assertTrue(np.array_equal(fold0_indexes, np.arange(50)))
        self.assertTrue(np.array_equal(fold1_indexes, np.arange(50, 100)))


if __name__ == "__main__":
    unittest.main()
