# Changelog

All notable changes to this project will be documented in this file.

## [10.20.0] - 2025-12-14

### Breaking Changes
- **Model Compatibility:** Due to the migration of the preprocessing engine from Python to C++, models saved with `aplr` versions `10.18.0` through `10.19.3` are not compatible with version `10.20.0` if they were trained on data that triggered the Python-based preprocessing (e.g., a `pandas.DataFrame` with categorical features or missing values). These older models may fail or produce unexpected results during prediction and must be retrained using version `10.20.0` or newer.

### Changed
- **Preprocessing Engine:** The entire data preprocessing pipeline, including missing value imputation and one-hot encoding, has been moved from Python to C++ for improved robustness. This change ensures that all data transformations are handled within the core engine.

### Added
- **`preprocess` Parameter:** A new boolean parameter `preprocess` has been added to the `APLRRegressor` and `APLRClassifier` constructors.
  - When `True` (default), the model automatically handles missing values and one-hot encodes categorical features in `pandas.DataFrame` inputs.
  - When `False`, automatic preprocessing is disabled. In this mode, the input `X` must be a purely numeric `numpy.ndarray` or `pandas.DataFrame`. This provides greater control for users who prefer to manage their own preprocessing pipelines and can result in performance gains and a lower memory footprint.

## [10.19.3] - 2025-12-07
### Fixed
- **Input Validation for `X_names`**: Resolved a `ValueError` that occurred when a NumPy array was passed to the `X_names` parameter in the `fit` method. The input validation has been enhanced to gracefully handle any list-like iterable (including NumPy arrays and tuples), ensuring robust and predictable behavior when providing feature names for non-DataFrame inputs.

## [10.19.2] - 2025-12-06
### Fixed
- **Memory Optimization:** Optimized the Python preprocessing pipeline to fix a bug that caused unnecessary memory consumption. A "just-in-time" copy mechanism now prevents large data copies during both fitting and prediction, improving memory safety for all input types.
- **Preprocessing Robustness:** Addressed a `RuntimeWarning: Mean of empty slice` that occurred during median imputation in `_preprocess_X_fit` when a column contained only missing values. The median calculation now gracefully handles such cases.
- **Backward Compatibility:** Improved `__setstate__` to safely load older pickled models by initializing new preprocessing attributes to their correct default types (e.g., `[]`), preventing `TypeError` exceptions.

### Changed
- **Code Quality and Maintainability:** The entire Python preprocessing pipeline was refactored into a clean `fit`/`transform` pattern. This separation of concerns removes boolean flags and significantly improves code clarity, making it easier to maintain and debug.
- **Type Hinting:** Added comprehensive type hints to all preprocessing methods for better readability and to enable static analysis.

### Documentation
- Updated API references and changelog for improved clarity and accuracy regarding the automatic preprocessing of `numpy.ndarray` and `pandas.DataFrame` inputs.

## [10.19.1] - 2025-11-22
### Fixed
- Improved thread safety in the parallel processing loop for estimating term split points.

## [10.19.0] - 2025-11-16

### Added
- Cross-validation results (predictions, y-values, sample weights, and observation indexes for each fold) are now stored in the fitted `APLRRegressor` model.
- This allows for more detailed post-training analysis of the cross-validation process.
- The CV results are persisted when the model is pickled.
- New methods in `APLRRegressor` to access this data:
  - `get_num_cv_folds()`
  - `get_cv_validation_predictions(fold_index)`
  - `get_cv_y(fold_index)`
  - `get_cv_sample_weight(fold_index)`
  - `get_cv_validation_indexes(fold_index)`
- New method `clear_cv_results()` in both `APLRRegressor` and `APLRClassifier` to manually clear the stored cross-validation data and free up memory.

### Fixed
- Corrected a bug in `APLRRegressor` where `sample_weight` was not handled correctly. The initialization (if not provided) and normalization of sample weights now occur on the full dataset before it is split into training and validation sets. This ensures consistent weight scaling across all cross-validation folds.

## [10.18.1] - 2025-10-30

### Fixed
- **Improved Backward Compatibility for Saved Models:** Resolved an issue where loading models trained with older versions of `aplr` would fail due to missing attributes. The `__setstate__` method now initializes new preprocessing-related attributes to `None` for older models, ensuring they can be loaded and used without `AttributeError` exceptions.
- **Stability for Unfitted Models:** Fixed a crash that occurred when calling `predict` on an unfitted `APLRClassifier`. The model now correctly raises a `RuntimeError` with an informative message in this scenario, improving stability and user feedback.
- **Restored Flexibility for `X_names` Parameter:** Fixed a regression from v10.18.0 where the `X_names` parameter no longer accepted `numpy.ndarray` or other list-like inputs. The parameter now correctly handles these types again, restoring flexibility for non-DataFrame inputs.

## [10.18.0] - 2025-10-29

### Added
- **Automatic Data Preprocessing with `pandas.DataFrame`**:
  - The model now automatically handles missing values and categorical features.
  - **Missing Value Imputation**: For both `numpy.ndarray` and `pandas.DataFrame` inputs, columns with missing values (`NaN`) are imputed using the column's sample weighted median. A new binary feature (e.g., `feature_name_missing`) is created to indicate where imputation occurred.
  - **Categorical Feature Encoding**: When a `pandas.DataFrame` is provided, columns with `object` or `category` data types are automatically one-hot encoded. The model gracefully handles unseen category levels during prediction by creating columns for all categories seen during training and setting those of them not seen during prediction to zero.

### Changed
- **Enhanced Flexibility in `APLRClassifier`**: The classifier now automatically converts numeric target arrays (e.g., `[0, 1, 0, 1]`) into string representations, simplifying setup for classification tasks.
- **Updated Documentation and Examples**: The API reference and examples have been updated to reflect the new automatic preprocessing capabilities.
