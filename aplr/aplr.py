from typing import List, Callable, Optional, Dict, Union
import numpy as np
import pandas as pd
import aplr_cpp
import itertools

FloatVector = np.ndarray
FloatMatrix = np.ndarray
IntVector = np.ndarray
IntMatrix = np.ndarray


class BaseAPLR:
    def _validate_X_fit_rows(self, X):
        """Checks if X has enough rows to be fitted."""
        if (isinstance(X, np.ndarray) and X.shape[0] < 2) or (
            isinstance(X, pd.DataFrame) and len(X) < 2
        ):
            raise ValueError("Input X must have at least 2 rows to be fitted.")

    def _common_X_preprocessing(self, X, is_fitting: bool, X_names=None):
        """Common preprocessing for fit and predict."""
        is_dataframe_input = isinstance(X, pd.DataFrame)

        if X_names is not None:
            X_names = list(X_names)

        if not is_dataframe_input:
            try:
                X_numeric = np.array(X, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    "Input X must be numeric if not a pandas DataFrame."
                ) from e
            X = pd.DataFrame(X_numeric)
            if is_fitting:
                if X_names:
                    X.columns = X_names
                else:
                    X.columns = [f"X{i}" for i in range(X.shape[1])]
            elif self.X_names_ and len(self.X_names_) == X.shape[1]:
                X.columns = self.X_names_
        else:  # X is already a DataFrame
            X = X.copy()  # Always copy to avoid modifying original
            if not is_fitting and self.X_names_:
                # Check if input columns for prediction match training columns (before OHE)
                if set(X.columns) != set(self.X_names_):
                    raise ValueError(
                        "Input columns for prediction do not match training columns."
                    )
                X = X[self.X_names_]  # Ensure order of original columns

        if is_fitting:
            self.X_names_ = list(X.columns)
            self.categorical_features_ = list(
                X.select_dtypes(include=["category", "object"]).columns
            )
            # Ensure it's an empty list if no categorical features, not None
            if not self.categorical_features_:
                self.categorical_features_ = []

        # Apply OHE if categorical_features_ were found during fitting.
        if self.categorical_features_:
            X = pd.get_dummies(X, columns=self.categorical_features_, dummy_na=False)
            if is_fitting:
                self.ohe_columns_ = list(X.columns)
                # Ensure it's an empty list if no OHE columns, not None
                if not self.ohe_columns_:
                    self.ohe_columns_ = []
            else:
                missing_cols = set(self.ohe_columns_) - set(X.columns)
                for c in missing_cols:
                    X[c] = 0
                X = X[self.ohe_columns_]  # Enforce column order

        if is_fitting:
            self.na_imputed_cols_ = [col for col in X.columns if X[col].isnull().any()]
            # Ensure it's an empty list if no NA imputed columns, not None
            if not self.na_imputed_cols_:
                self.na_imputed_cols_ = []

        # Apply NA indicator if na_imputed_cols_ were found during fitting.
        if self.na_imputed_cols_:
            for col in self.na_imputed_cols_:
                X[col + "_missing"] = X[col].isnull().astype(int)

        if not is_fitting and self.median_values_:
            for col in self.median_values_:  # Iterate over keys if it's a dict
                if col in X.columns:
                    X[col] = X[col].fillna(self.median_values_[col])

        return X

    def _preprocess_X_fit(self, X, X_names, sample_weight):
        if sample_weight.size > 0:
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be a 1D array.")
            if len(sample_weight) != X.shape[0]:
                raise ValueError(
                    "sample_weight must have the same number of rows as X."
                )
            if np.any(np.isnan(sample_weight)) or np.any(np.isinf(sample_weight)):
                raise ValueError("sample_weight cannot contain nan or infinite values.")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative values.")
        X = self._common_X_preprocessing(X, is_fitting=True, X_names=X_names)
        self.median_values_ = {}
        numeric_cols_for_median = [col for col in X.columns if "_missing" not in col]
        for col in numeric_cols_for_median:
            missing_mask = X[col].isnull()

            if sample_weight.size > 0:
                valid_indices = ~missing_mask
                col_data = X.loc[valid_indices, col]
                col_weights = sample_weight[valid_indices]
                if col_data.empty:
                    median_val = 0
                else:
                    col_data_np = col_data.to_numpy()
                    sort_indices = np.argsort(col_data_np, kind="stable")
                    sorted_data = col_data_np[sort_indices]
                    sorted_weights = col_weights[sort_indices]

                    cumulative_weights = np.cumsum(sorted_weights)
                    total_weight = cumulative_weights[-1]

                    median_weight_index = np.searchsorted(
                        cumulative_weights, total_weight / 2.0
                    )
                    if median_weight_index >= len(sorted_data):
                        median_weight_index = len(sorted_data) - 1
                    median_val = sorted_data[median_weight_index]
            else:
                median_val = X[col].median()

            if pd.isna(median_val):
                median_val = 0

            self.median_values_[col] = median_val
            X[col] = X[col].fillna(median_val)

        self.final_training_columns_ = list(X.columns)
        return X.values.astype(np.float64), list(X.columns)

    def _preprocess_X_predict(self, X):
        X = self._common_X_preprocessing(X, is_fitting=False)

        # Enforce column order from training if it was set.
        if self.final_training_columns_:
            X = X[self.final_training_columns_]

        return X.values.astype(np.float64)

    def __setstate__(self, state):
        """Handles unpickling for backward compatibility."""
        self.__dict__.update(state)

        # For backward compatibility, initialize new attributes to None if they don't exist,
        # indicating the model was trained before these features were introduced.
        new_attributes = [
            "X_names_",
            "categorical_features_",
            "ohe_columns_",
            "na_imputed_cols_",
            "median_values_",
            "final_training_columns_",
        ]
        for attr in new_attributes:
            if not hasattr(self, attr):
                setattr(self, attr, None)


class APLRRegressor(BaseAPLR):
    def __init__(
        self,
        m: int = 3000,
        v: float = 0.5,
        random_state: int = 0,
        loss_function: str = "mse",
        link_function: str = "identity",
        n_jobs: int = 0,
        cv_folds: int = 5,
        bins: int = 300,
        max_interaction_level: int = 1,
        max_interactions: int = 100000,
        min_observations_in_split: int = 4,
        ineligible_boosting_steps_added: int = 15,
        max_eligible_terms: int = 7,
        verbosity: int = 0,
        dispersion_parameter: float = 1.5,
        validation_tuning_metric: str = "default",
        quantile: float = 0.5,
        calculate_custom_validation_error_function: Optional[
            Callable[
                [
                    FloatVector,
                    FloatVector,
                    FloatVector,
                    FloatVector,
                    FloatMatrix,
                ],
                float,
            ]
        ] = None,
        calculate_custom_loss_function: Optional[
            Callable[
                [
                    FloatVector,
                    FloatVector,
                    FloatVector,
                    FloatVector,
                    FloatMatrix,
                ],
                float,
            ]
        ] = None,
        calculate_custom_negative_gradient_function: Optional[
            Callable[
                [FloatVector, FloatVector, FloatVector, FloatMatrix],
                FloatVector,
            ]
        ] = None,
        calculate_custom_transform_linear_predictor_to_predictions_function: Optional[
            Callable[[FloatVector], FloatVector]
        ] = None,
        calculate_custom_differentiate_predictions_wrt_linear_predictor_function: Optional[
            Callable[[FloatVector], FloatVector]
        ] = None,
        boosting_steps_before_interactions_are_allowed: int = 0,
        monotonic_constraints_ignore_interactions: bool = False,
        group_mse_by_prediction_bins: int = 10,
        group_mse_cycle_min_obs_in_bin: int = 30,
        early_stopping_rounds: int = 200,
        num_first_steps_with_linear_effects_only: int = 0,
        penalty_for_non_linearity: float = 0.0,
        penalty_for_interactions: float = 0.0,
        max_terms: int = 0,
        ridge_penalty: float = 0.0001,
        mean_bias_correction: bool = False,
        faster_convergence: bool = False,
    ):
        self.m = m
        self.v = v
        self.random_state = random_state
        self.loss_function = loss_function
        self.link_function = link_function
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.bins = bins
        self.max_interaction_level = max_interaction_level
        self.max_interactions = max_interactions
        self.min_observations_in_split = min_observations_in_split
        self.ineligible_boosting_steps_added = ineligible_boosting_steps_added
        self.max_eligible_terms = max_eligible_terms
        self.verbosity = verbosity
        self.dispersion_parameter = dispersion_parameter
        self.validation_tuning_metric = validation_tuning_metric
        self.quantile = quantile
        self.calculate_custom_validation_error_function = (
            calculate_custom_validation_error_function
        )
        self.calculate_custom_loss_function = calculate_custom_loss_function
        self.calculate_custom_negative_gradient_function = (
            calculate_custom_negative_gradient_function
        )
        self.calculate_custom_transform_linear_predictor_to_predictions_function = (
            calculate_custom_transform_linear_predictor_to_predictions_function
        )
        self.calculate_custom_differentiate_predictions_wrt_linear_predictor_function = (
            calculate_custom_differentiate_predictions_wrt_linear_predictor_function
        )
        self.boosting_steps_before_interactions_are_allowed = (
            boosting_steps_before_interactions_are_allowed
        )
        self.monotonic_constraints_ignore_interactions = (
            monotonic_constraints_ignore_interactions
        )
        self.group_mse_by_prediction_bins = group_mse_by_prediction_bins
        self.group_mse_cycle_min_obs_in_bin = group_mse_cycle_min_obs_in_bin
        self.early_stopping_rounds = early_stopping_rounds
        self.num_first_steps_with_linear_effects_only = (
            num_first_steps_with_linear_effects_only
        )
        self.penalty_for_non_linearity = penalty_for_non_linearity
        self.penalty_for_interactions = penalty_for_interactions
        self.max_terms = max_terms
        self.ridge_penalty = ridge_penalty
        self.mean_bias_correction = mean_bias_correction
        self.faster_convergence = faster_convergence

        # Data transformations
        self.median_values_ = {}
        self.categorical_features_ = []
        self.ohe_columns_ = []
        self.na_imputed_cols_ = []
        self.X_names_ = []
        self.final_training_columns_ = []

        # Creating aplr_cpp and setting parameters
        self.APLRRegressor = aplr_cpp.APLRRegressor()
        self.__set_params_cpp()

    # Sets parameters for aplr_cpp.APLRRegressor cpp object
    def __set_params_cpp(self):
        self.APLRRegressor.m = self.m
        self.APLRRegressor.v = self.v
        self.APLRRegressor.random_state = self.random_state
        self.APLRRegressor.loss_function = self.loss_function
        self.APLRRegressor.link_function = self.link_function
        self.APLRRegressor.n_jobs = self.n_jobs
        self.APLRRegressor.cv_folds = self.cv_folds
        self.APLRRegressor.bins = self.bins
        self.APLRRegressor.max_interaction_level = self.max_interaction_level
        self.APLRRegressor.max_interactions = self.max_interactions
        self.APLRRegressor.min_observations_in_split = self.min_observations_in_split
        self.APLRRegressor.ineligible_boosting_steps_added = (
            self.ineligible_boosting_steps_added
        )
        self.APLRRegressor.max_eligible_terms = self.max_eligible_terms
        self.APLRRegressor.verbosity = self.verbosity
        self.APLRRegressor.dispersion_parameter = self.dispersion_parameter
        self.APLRRegressor.validation_tuning_metric = self.validation_tuning_metric
        self.APLRRegressor.quantile = self.quantile
        self.APLRRegressor.calculate_custom_validation_error_function = (
            self.calculate_custom_validation_error_function
        )
        self.APLRRegressor.calculate_custom_loss_function = (
            self.calculate_custom_loss_function
        )
        self.APLRRegressor.calculate_custom_negative_gradient_function = (
            self.calculate_custom_negative_gradient_function
        )
        self.APLRRegressor.calculate_custom_transform_linear_predictor_to_predictions_function = (
            self.calculate_custom_transform_linear_predictor_to_predictions_function
        )
        self.APLRRegressor.calculate_custom_differentiate_predictions_wrt_linear_predictor_function = (
            self.calculate_custom_differentiate_predictions_wrt_linear_predictor_function
        )
        self.APLRRegressor.boosting_steps_before_interactions_are_allowed = (
            self.boosting_steps_before_interactions_are_allowed
        )
        self.APLRRegressor.monotonic_constraints_ignore_interactions = (
            self.monotonic_constraints_ignore_interactions
        )
        self.APLRRegressor.group_mse_by_prediction_bins = (
            self.group_mse_by_prediction_bins
        )
        self.APLRRegressor.group_mse_cycle_min_obs_in_bin = (
            self.group_mse_cycle_min_obs_in_bin
        )
        self.APLRRegressor.early_stopping_rounds = self.early_stopping_rounds
        self.APLRRegressor.num_first_steps_with_linear_effects_only = (
            self.num_first_steps_with_linear_effects_only
        )
        self.APLRRegressor.penalty_for_non_linearity = self.penalty_for_non_linearity
        self.APLRRegressor.penalty_for_interactions = self.penalty_for_interactions
        self.APLRRegressor.max_terms = self.max_terms
        self.APLRRegressor.ridge_penalty = self.ridge_penalty
        self.APLRRegressor.mean_bias_correction = self.mean_bias_correction
        self.APLRRegressor.faster_convergence = self.faster_convergence

    def fit(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        y: FloatVector,
        sample_weight: FloatVector = np.empty(0),
        X_names: List[str] = [],
        cv_observations: IntMatrix = np.empty([0, 0]),
        prioritized_predictors_indexes: List[int] = [],
        monotonic_constraints: List[int] = [],
        group: FloatVector = np.empty(0),
        interaction_constraints: List[List[int]] = [],
        other_data: FloatMatrix = np.empty([0, 0]),
        predictor_learning_rates: List[float] = [],
        predictor_penalties_for_non_linearity: List[float] = [],
        predictor_penalties_for_interactions: List[float] = [],
        predictor_min_observations_in_split: List[int] = [],
    ):
        self._validate_X_fit_rows(X)
        self.__set_params_cpp()
        X_transformed, X_names_transformed = self._preprocess_X_fit(
            X, X_names, sample_weight
        )
        self.APLRRegressor.fit(
            X_transformed,
            y,
            sample_weight,
            X_names_transformed,
            cv_observations,
            prioritized_predictors_indexes,
            monotonic_constraints,
            group,
            interaction_constraints,
            other_data,
            predictor_learning_rates,
            predictor_penalties_for_non_linearity,
            predictor_penalties_for_interactions,
            predictor_min_observations_in_split,
        )

    def predict(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        cap_predictions_to_minmax_in_training: bool = True,
    ) -> FloatVector:
        if self.link_function == "custom_function":
            self.APLRRegressor.calculate_custom_transform_linear_predictor_to_predictions_function = (
                self.calculate_custom_transform_linear_predictor_to_predictions_function
            )
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRRegressor.predict(
            X_transformed, cap_predictions_to_minmax_in_training
        )

    def set_term_names(self, X_names: List[str]):
        self.APLRRegressor.set_term_names(X_names)

    def calculate_feature_importance(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        sample_weight: FloatVector = np.empty(0),
    ) -> FloatVector:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRRegressor.calculate_feature_importance(
            X_transformed, sample_weight
        )

    def calculate_term_importance(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        sample_weight: FloatVector = np.empty(0),
    ) -> FloatVector:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRRegressor.calculate_term_importance(
            X_transformed, sample_weight
        )

    def calculate_local_feature_contribution(
        self, X: Union[pd.DataFrame, FloatMatrix]
    ) -> FloatMatrix:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRRegressor.calculate_local_feature_contribution(X_transformed)

    def calculate_local_term_contribution(
        self, X: Union[pd.DataFrame, FloatMatrix]
    ) -> FloatMatrix:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRRegressor.calculate_local_term_contribution(X_transformed)

    def calculate_local_contribution_from_selected_terms(
        self, X: Union[pd.DataFrame, FloatMatrix], predictor_indexes: List[int]
    ) -> FloatVector:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRRegressor.calculate_local_contribution_from_selected_terms(
            X_transformed, predictor_indexes
        )

    def calculate_terms(self, X: Union[pd.DataFrame, FloatMatrix]) -> FloatMatrix:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRRegressor.calculate_terms(X_transformed)

    def get_term_names(self) -> List[str]:
        return self.APLRRegressor.get_term_names()

    def get_term_affiliations(self) -> List[str]:
        return self.APLRRegressor.get_term_affiliations()

    def get_unique_term_affiliations(self) -> List[str]:
        return self.APLRRegressor.get_unique_term_affiliations()

    def get_base_predictors_in_each_unique_term_affiliation(self) -> List[List[int]]:
        return self.APLRRegressor.get_base_predictors_in_each_unique_term_affiliation()

    def get_term_coefficients(self) -> FloatVector:
        return self.APLRRegressor.get_term_coefficients()

    def get_validation_error_steps(self) -> FloatMatrix:
        return self.APLRRegressor.get_validation_error_steps()

    def get_feature_importance(self) -> FloatVector:
        return self.APLRRegressor.get_feature_importance()

    def get_term_importance(self) -> FloatVector:
        return self.APLRRegressor.get_term_importance()

    def get_term_main_predictor_indexes(self) -> IntVector:
        return self.APLRRegressor.get_term_main_predictor_indexes()

    def get_term_interaction_levels(self) -> IntVector:
        return self.APLRRegressor.get_term_interaction_levels()

    def get_intercept(self) -> float:
        return self.APLRRegressor.get_intercept()

    def get_optimal_m(self) -> int:
        return self.APLRRegressor.get_optimal_m()

    def get_validation_tuning_metric(self) -> str:
        return self.APLRRegressor.get_validation_tuning_metric()

    def get_main_effect_shape(self, predictor_index: int) -> Dict[float, float]:
        return self.APLRRegressor.get_main_effect_shape(predictor_index)

    def get_unique_term_affiliation_shape(
        self,
        unique_term_affiliation: str,
        max_rows_before_sampling: int = 500000,
        additional_points: int = 250,
    ) -> FloatMatrix:
        return self.APLRRegressor.get_unique_term_affiliation_shape(
            unique_term_affiliation, max_rows_before_sampling, additional_points
        )

    def get_cv_error(self) -> float:
        return self.APLRRegressor.get_cv_error()

    def get_num_cv_folds(self) -> int:
        """
        Gets the number of cross-validation folds used during training.

        :return: The number of folds.
        """
        return self.APLRRegressor.get_num_cv_folds()

    def get_cv_validation_predictions(self, fold_index: int) -> FloatVector:
        """
        Gets the validation predictions for a specific cross-validation fold.

        Note that these predictions may be conservative, as the final model is an ensemble of the models
        from all cross-validation folds, which has a variance-reducing effect similar to bagging.

        :param fold_index: The index of the fold.
        :return: A numpy array containing the validation predictions.
        """
        return self.APLRRegressor.get_cv_validation_predictions(fold_index)

    def get_cv_y(self, fold_index: int) -> FloatVector:
        """
        Gets the validation response values (y) for a specific cross-validation fold.

        :param fold_index: The index of the fold.
        :return: A numpy array containing the validation response values.
        """
        return self.APLRRegressor.get_cv_y(fold_index)

    def get_cv_sample_weight(self, fold_index: int) -> FloatVector:
        """
        Gets the validation sample weights for a specific cross-validation fold.

        :param fold_index: The index of the fold.
        :return: A numpy array containing the validation sample weights.
        """
        return self.APLRRegressor.get_cv_sample_weight(fold_index)

    def get_cv_validation_indexes(self, fold_index: int) -> IntVector:
        """
        Gets the original indexes of the validation observations for a specific cross-validation fold.

        :param fold_index: The index of the fold.
        :return: A numpy array containing the original indexes.
        """
        return self.APLRRegressor.get_cv_validation_indexes(fold_index)

    def set_intercept(self, value: float):
        self.APLRRegressor.set_intercept(value)

    def plot_affiliation_shape(
        self,
        affiliation: str,
        plot: bool = True,
        save: bool = False,
        path: str = "",
    ):
        """
        Plots or saves the shape of a given unique term affiliation.

        For main effects, it produces a line plot. For two-way interactions, it produces a heatmap.
        Plotting for higher-order interactions is not supported.

        :param affiliation: A string specifying which unique_term_affiliation to use.
        :param plot: If True, displays the plot.
        :param save: If True, saves the plot to a file.
        :param path: The file path to save the plot. If empty and save is True, a default path will be used.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Please install it.")

        all_affiliations = self.get_unique_term_affiliations()
        if affiliation not in all_affiliations:
            raise ValueError(
                f"Affiliation '{affiliation}' not found in model. "
                f"Available affiliations are: {all_affiliations}"
            )

        affiliation_index = all_affiliations.index(affiliation)

        predictors_in_each_affiliation = (
            self.get_base_predictors_in_each_unique_term_affiliation()
        )
        predictor_indexes_used = predictors_in_each_affiliation[affiliation_index]

        shape = self.get_unique_term_affiliation_shape(affiliation)
        if shape.shape[0] == 0:
            print(f"No shape data available for affiliation '{affiliation}'.")
            return

        predictor_names = affiliation.split(" & ")

        is_main_effect: bool = len(predictor_indexes_used) == 1
        is_two_way_interaction: bool = len(predictor_indexes_used) == 2

        if is_main_effect:
            fig = plt.figure()
            # Sort by predictor value for a clean line plot
            sorted_indices = np.argsort(shape[:, 0])
            plt.plot(shape[sorted_indices, 0], shape[sorted_indices, 1])
            plt.xlabel(predictor_names[0])
            plt.ylabel("Contribution to linear predictor")
            plt.title(f"Main effect of {predictor_names[0]}")
            plt.grid(True)
        elif is_two_way_interaction:
            fig = plt.figure(figsize=(8, 6))

            # Get unique coordinates and their inverse mapping
            y_unique, y_inv = np.unique(shape[:, 0], return_inverse=True)
            x_unique, x_inv = np.unique(shape[:, 1], return_inverse=True)

            # Create grid for sums and counts
            grid_sums = np.zeros((len(y_unique), len(x_unique)))
            grid_counts = np.zeros((len(y_unique), len(x_unique)))

            # Populate sums and counts to later calculate the mean
            np.add.at(grid_sums, (y_inv, x_inv), shape[:, 2])
            np.add.at(grid_counts, (y_inv, x_inv), 1)

            # Calculate mean, avoiding division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                pivot_table_values = np.true_divide(grid_sums, grid_counts)
                # Where there's no data, pivot_table_values will be nan, which is fine for imshow.

            plt.imshow(
                pivot_table_values,
                aspect="auto",
                origin="lower",
                extent=[
                    x_unique.min(),
                    x_unique.max(),
                    y_unique.min(),
                    y_unique.max(),
                ],
                cmap="Blues_r",
            )
            plt.colorbar(label="Contribution to the linear predictor")
            plt.xlabel(predictor_names[1])
            plt.ylabel(predictor_names[0])
            plt.title(
                f"Interaction between {predictor_names[0]} and {predictor_names[1]}"
            )
        else:
            print(
                f"Plotting for interaction level > 2 is not supported. Affiliation: {affiliation}"
            )
            return

        if save:
            save_path = path or f"shape_of_{affiliation.replace(' & ', '_')}.png"
            plt.savefig(save_path)

        if plot:
            plt.show()

        plt.close(fig)

    def remove_provided_custom_functions(self):
        self.APLRRegressor.remove_provided_custom_functions()
        self.calculate_custom_validation_error_function = None
        self.calculate_custom_loss_function = None
        self.calculate_custom_negative_gradient_function = None

    def clear_cv_results(self):
        """
        Clears the stored cross-validation results (predictions, y, etc.) to free up memory.
        """
        self.APLRRegressor.clear_cv_results()

    # For sklearn
    def get_params(self, deep=True):
        return {
            "m": self.m,
            "v": self.v,
            "random_state": self.random_state,
            "loss_function": self.loss_function,
            "link_function": self.link_function,
            "n_jobs": self.n_jobs,
            "cv_folds": self.cv_folds,
            "bins": self.bins,
            "max_interaction_level": self.max_interaction_level,
            "max_interactions": self.max_interactions,
            "verbosity": self.verbosity,
            "min_observations_in_split": self.min_observations_in_split,
            "ineligible_boosting_steps_added": self.ineligible_boosting_steps_added,
            "max_eligible_terms": self.max_eligible_terms,
            "dispersion_parameter": self.dispersion_parameter,
            "validation_tuning_metric": self.validation_tuning_metric,
            "quantile": self.quantile,
            "calculate_custom_validation_error_function": self.calculate_custom_validation_error_function,
            "calculate_custom_loss_function": self.calculate_custom_loss_function,
            "calculate_custom_negative_gradient_function": self.calculate_custom_negative_gradient_function,
            "calculate_custom_transform_linear_predictor_to_predictions_function": self.calculate_custom_transform_linear_predictor_to_predictions_function,
            "calculate_custom_differentiate_predictions_wrt_linear_predictor_function": self.calculate_custom_differentiate_predictions_wrt_linear_predictor_function,
            "boosting_steps_before_interactions_are_allowed": self.boosting_steps_before_interactions_are_allowed,
            "monotonic_constraints_ignore_interactions": self.monotonic_constraints_ignore_interactions,
            "group_mse_by_prediction_bins": self.group_mse_by_prediction_bins,
            "group_mse_cycle_min_obs_in_bin": self.group_mse_cycle_min_obs_in_bin,
            "early_stopping_rounds": self.early_stopping_rounds,
            "num_first_steps_with_linear_effects_only": self.num_first_steps_with_linear_effects_only,
            "penalty_for_non_linearity": self.penalty_for_non_linearity,
            "penalty_for_interactions": self.penalty_for_interactions,
            "max_terms": self.max_terms,
            "ridge_penalty": self.ridge_penalty,
            "mean_bias_correction": self.mean_bias_correction,
            "faster_convergence": self.faster_convergence,
        }

    # For sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__set_params_cpp()
        return self


class APLRClassifier(BaseAPLR):
    def __init__(
        self,
        m: int = 3000,
        v: float = 0.5,
        random_state: int = 0,
        n_jobs: int = 0,
        cv_folds: int = 5,
        bins: int = 300,
        verbosity: int = 0,
        max_interaction_level: int = 1,
        max_interactions: int = 100000,
        min_observations_in_split: int = 4,
        ineligible_boosting_steps_added: int = 15,
        max_eligible_terms: int = 7,
        boosting_steps_before_interactions_are_allowed: int = 0,
        monotonic_constraints_ignore_interactions: bool = False,
        early_stopping_rounds: int = 200,
        num_first_steps_with_linear_effects_only: int = 0,
        penalty_for_non_linearity: float = 0.0,
        penalty_for_interactions: float = 0.0,
        max_terms: int = 0,
        ridge_penalty: float = 0.0001,
    ):
        self.m = m
        self.v = v
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.bins = bins
        self.verbosity = verbosity
        self.max_interaction_level = max_interaction_level
        self.max_interactions = max_interactions
        self.min_observations_in_split = min_observations_in_split
        self.ineligible_boosting_steps_added = ineligible_boosting_steps_added
        self.max_eligible_terms = max_eligible_terms
        self.boosting_steps_before_interactions_are_allowed = (
            boosting_steps_before_interactions_are_allowed
        )
        self.monotonic_constraints_ignore_interactions = (
            monotonic_constraints_ignore_interactions
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.num_first_steps_with_linear_effects_only = (
            num_first_steps_with_linear_effects_only
        )
        self.penalty_for_non_linearity = penalty_for_non_linearity
        self.penalty_for_interactions = penalty_for_interactions
        self.max_terms = max_terms
        self.ridge_penalty = ridge_penalty

        # Data transformations
        self.median_values_ = {}
        self.categorical_features_ = []
        self.ohe_columns_ = []
        self.na_imputed_cols_ = []
        self.X_names_ = []
        self.final_training_columns_ = []

        # Creating aplr_cpp and setting parameters
        self.APLRClassifier = aplr_cpp.APLRClassifier()
        self.__set_params_cpp()

    # Sets parameters for aplr_cpp.APLRClassifier cpp object
    def __set_params_cpp(self):
        self.APLRClassifier.m = self.m
        self.APLRClassifier.v = self.v
        self.APLRClassifier.random_state = self.random_state
        self.APLRClassifier.n_jobs = self.n_jobs
        self.APLRClassifier.cv_folds = self.cv_folds
        self.APLRClassifier.bins = self.bins
        self.APLRClassifier.verbosity = self.verbosity
        self.APLRClassifier.max_interaction_level = self.max_interaction_level
        self.APLRClassifier.max_interactions = self.max_interactions
        self.APLRClassifier.min_observations_in_split = self.min_observations_in_split
        self.APLRClassifier.ineligible_boosting_steps_added = (
            self.ineligible_boosting_steps_added
        )
        self.APLRClassifier.max_eligible_terms = self.max_eligible_terms
        self.APLRClassifier.boosting_steps_before_interactions_are_allowed = (
            self.boosting_steps_before_interactions_are_allowed
        )
        self.APLRClassifier.monotonic_constraints_ignore_interactions = (
            self.monotonic_constraints_ignore_interactions
        )
        self.APLRClassifier.early_stopping_rounds = self.early_stopping_rounds
        self.APLRClassifier.num_first_steps_with_linear_effects_only = (
            self.num_first_steps_with_linear_effects_only
        )
        self.APLRClassifier.penalty_for_non_linearity = self.penalty_for_non_linearity
        self.APLRClassifier.penalty_for_interactions = self.penalty_for_interactions
        self.APLRClassifier.max_terms = self.max_terms
        self.APLRClassifier.ridge_penalty = self.ridge_penalty

    def fit(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        y: Union[FloatVector, List[str]],
        sample_weight: FloatVector = np.empty(0),
        X_names: List[str] = [],
        cv_observations: IntMatrix = np.empty([0, 0]),
        prioritized_predictors_indexes: List[int] = [],
        monotonic_constraints: List[int] = [],
        interaction_constraints: List[List[int]] = [],
        predictor_learning_rates: List[float] = [],
        predictor_penalties_for_non_linearity: List[float] = [],
        predictor_penalties_for_interactions: List[float] = [],
        predictor_min_observations_in_split: List[int] = [],
    ):
        self._validate_X_fit_rows(X)
        self.__set_params_cpp()
        X_transformed, X_names_transformed = self._preprocess_X_fit(
            X, X_names, sample_weight
        )

        if isinstance(y, np.ndarray):
            y = y.astype(str).tolist()
        elif isinstance(y, list) and y and not isinstance(y[0], str):
            y = [str(val) for val in y]

        self.APLRClassifier.fit(
            X_transformed,
            y,
            sample_weight,
            X_names_transformed,
            cv_observations,
            prioritized_predictors_indexes,
            monotonic_constraints,
            interaction_constraints,
            predictor_learning_rates,
            predictor_penalties_for_non_linearity,
            predictor_penalties_for_interactions,
            predictor_min_observations_in_split,
        )
        # For sklearn
        self.classes_ = np.arange(len(self.APLRClassifier.get_categories()))

    def predict_class_probabilities(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        cap_predictions_to_minmax_in_training: bool = False,
    ) -> FloatMatrix:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRClassifier.predict_class_probabilities(
            X_transformed, cap_predictions_to_minmax_in_training
        )

    def predict(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        cap_predictions_to_minmax_in_training: bool = False,
    ) -> List[str]:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRClassifier.predict(
            X_transformed, cap_predictions_to_minmax_in_training
        )

    def calculate_local_feature_contribution(
        self, X: Union[pd.DataFrame, FloatMatrix]
    ) -> FloatMatrix:
        X_transformed = self._preprocess_X_predict(X)
        return self.APLRClassifier.calculate_local_feature_contribution(X_transformed)

    def get_categories(self) -> List[str]:
        return self.APLRClassifier.get_categories()

    def get_logit_model(self, category: str) -> APLRRegressor:
        logit_model_cpp = self.APLRClassifier.get_logit_model(category)

        logit_model_py = APLRRegressor(
            m=self.m,
            v=self.v,
            random_state=self.random_state,
            loss_function="binomial",
            link_function="logit",
            n_jobs=self.n_jobs,
            cv_folds=self.cv_folds,
            bins=self.bins,
            max_interaction_level=self.max_interaction_level,
            max_interactions=self.max_interactions,
            min_observations_in_split=self.min_observations_in_split,
            ineligible_boosting_steps_added=self.ineligible_boosting_steps_added,
            max_eligible_terms=self.max_eligible_terms,
            verbosity=self.verbosity,
            boosting_steps_before_interactions_are_allowed=self.boosting_steps_before_interactions_are_allowed,
            monotonic_constraints_ignore_interactions=self.monotonic_constraints_ignore_interactions,
            early_stopping_rounds=self.early_stopping_rounds,
            num_first_steps_with_linear_effects_only=self.num_first_steps_with_linear_effects_only,
            penalty_for_non_linearity=self.penalty_for_non_linearity,
            penalty_for_interactions=self.penalty_for_interactions,
            max_terms=self.max_terms,
            ridge_penalty=self.ridge_penalty,
        )

        logit_model_py.APLRRegressor = logit_model_cpp

        return logit_model_py

    def get_validation_error_steps(self) -> FloatMatrix:
        return self.APLRClassifier.get_validation_error_steps()

    def get_cv_error(self) -> float:
        return self.APLRClassifier.get_cv_error()

    def get_feature_importance(self) -> FloatVector:
        return self.APLRClassifier.get_feature_importance()

    def get_unique_term_affiliations(self) -> List[str]:
        return self.APLRClassifier.get_unique_term_affiliations()

    def get_base_predictors_in_each_unique_term_affiliation(self) -> List[List[int]]:
        return self.APLRClassifier.get_base_predictors_in_each_unique_term_affiliation()

    def clear_cv_results(self):
        """
        Clears the stored cross-validation results from all underlying logit models to free up memory.
        """
        self.APLRClassifier.clear_cv_results()

    # For sklearn
    def get_params(self, deep=True):
        return {
            "m": self.m,
            "v": self.v,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "cv_folds": self.cv_folds,
            "bins": self.bins,
            "verbosity": self.verbosity,
            "max_interaction_level": self.max_interaction_level,
            "max_interactions": self.max_interactions,
            "min_observations_in_split": self.min_observations_in_split,
            "ineligible_boosting_steps_added": self.ineligible_boosting_steps_added,
            "max_eligible_terms": self.max_eligible_terms,
            "boosting_steps_before_interactions_are_allowed": self.boosting_steps_before_interactions_are_allowed,
            "monotonic_constraints_ignore_interactions": self.monotonic_constraints_ignore_interactions,
            "early_stopping_rounds": self.early_stopping_rounds,
            "num_first_steps_with_linear_effects_only": self.num_first_steps_with_linear_effects_only,
            "penalty_for_non_linearity": self.penalty_for_non_linearity,
            "penalty_for_interactions": self.penalty_for_interactions,
            "max_terms": self.max_terms,
            "ridge_penalty": self.ridge_penalty,
        }

    # For sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__set_params_cpp()
        return self

    # For sklearn
    def predict_proba(self, X: FloatMatrix) -> FloatMatrix:
        return self.predict_class_probabilities(X)


class APLRTuner:
    def __init__(
        self,
        parameters: Union[Dict[str, List[float]], List[Dict[str, List[float]]]] = {
            "max_interaction_level": [0, 1],
            "min_observations_in_split": [4, 10, 20, 100, 500, 1000],
        },
        is_regressor: bool = True,
    ):
        self.parameters = parameters
        self.is_regressor = is_regressor
        self.parameter_grid = self._create_parameter_grid()

    def _create_parameter_grid(self) -> List[Dict[str, float]]:
        items = sorted(self.parameters.items())
        keys, values = zip(*items)
        combinations = list(itertools.product(*values))
        grid = [dict(zip(keys, combination)) for combination in combinations]
        return grid

    def fit(self, X: Union[pd.DataFrame, FloatMatrix], y: FloatVector, **kwargs):
        self.cv_results: List[Dict[str, float]] = []
        best_validation_result = np.inf
        for params in self.parameter_grid:
            if self.is_regressor:
                model = APLRRegressor(**params)
            else:
                model = APLRClassifier(**params)
            model.fit(X, y, **kwargs)
            cv_error_for_this_model = model.get_cv_error()
            cv_results_for_this_model = model.get_params()
            cv_results_for_this_model["cv_error"] = cv_error_for_this_model
            self.cv_results.append(cv_results_for_this_model)
            if cv_error_for_this_model < best_validation_result:
                best_validation_result = cv_error_for_this_model
                self.best_model = model
        self.cv_results = sorted(self.cv_results, key=lambda x: x["cv_error"])

    def predict(
        self, X: Union[pd.DataFrame, FloatMatrix], **kwargs
    ) -> Union[FloatVector, List[str]]:
        return self.best_model.predict(X, **kwargs)

    def predict_class_probabilities(
        self, X: Union[pd.DataFrame, FloatMatrix], **kwargs
    ) -> FloatMatrix:
        if self.is_regressor == False:
            return self.best_model.predict_class_probabilities(X, **kwargs)
        else:
            raise TypeError(
                "predict_class_probabilities is only possible when is_regressor is False"
            )

    def predict_proba(
        self, X: Union[pd.DataFrame, FloatMatrix], **kwargs
    ) -> FloatMatrix:
        return self.predict_class_probabilities(X, **kwargs)

    def get_best_estimator(self) -> Union[APLRClassifier, APLRRegressor]:
        return self.best_model

    def get_cv_results(self) -> List[Dict[str, float]]:
        return self.cv_results
