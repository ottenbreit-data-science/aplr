# Changelog

All notable changes to this project will be documented in this file.

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
