import unittest
import pandas as pd
import numpy as np
from aplr import APLRClassifier, APLRRegressor
from aplr.aplr import _dataframe_to_cpp_dataframe


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

        # 5. Test fitting with a DataFrame containing non-numeric data (should fail)
        model = APLRRegressor(preprocess=False, random_state=0)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot convert DataFrame to matrix if it contains non-numeric columns. "
            "Please ensure all columns are numeric or set preprocess=True.",
        ):
            model.fit(X_train_df_mixed, y_single)


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
