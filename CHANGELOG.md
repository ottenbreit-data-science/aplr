# Changelog

All notable changes to this project will be documented in this file.

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
  - When a `pandas.DataFrame` is passed as input `X`, the model now automatically handles missing values and categorical features.
  - **Missing Value Imputation**: Columns with missing values (`NaN`) are imputed using the column's median. A new binary feature (e.g., `feature_name_missing`) is created to indicate where imputation occurred. The median calculation correctly handles `sample_weight`.
  - **Categorical Feature Encoding**: Columns with `object` or `category` data types are automatically one-hot encoded. The model gracefully handles unseen category levels during prediction by creating columns for all categories seen during training and setting those of them not seen during prediction to zero.

### Changed
- **Enhanced Flexibility in `APLRClassifier`**: The classifier now automatically converts numeric target arrays (e.g., `[0, 1, 0, 1]`) into string representations, simplifying setup for classification tasks.
- **Updated Documentation and Examples**: The API reference and examples have been updated to reflect the new automatic preprocessing capabilities.
