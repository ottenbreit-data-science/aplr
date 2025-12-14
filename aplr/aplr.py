from typing import List, Callable, Optional, Dict, Union
import numpy as np
import pandas as pd
import aplr_cpp
import itertools

FloatVector = np.ndarray
FloatMatrix = np.ndarray
IntVector = np.ndarray
IntMatrix = np.ndarray


def _dataframe_to_cpp_dataframe(df: pd.DataFrame) -> aplr_cpp.CppDataFrame:
    """Converts a pandas DataFrame to a CppDataFrame."""
    cpp_df = aplr_cpp.CppDataFrame()
    for col_name in df.columns:
        col = df[col_name]
        if pd.api.types.is_numeric_dtype(col.dtype):
            # Convert numeric columns to std::vector<double>
            # NaNs are preserved and handled in C++
            cpp_df.add_numeric_column(
                col_name, col.to_numpy(dtype=np.float64, na_value=np.nan)
            )
        elif (
            isinstance(col.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(col.dtype)
            or pd.api.types.is_string_dtype(col.dtype)
        ):
            # Convert categorical/object/string columns to std::vector<std::string>
            # Missing values (None, np.nan) are converted to empty strings for C++ handling
            cpp_df.add_categorical_column(col_name, col.astype(str).fillna("").tolist())
        else:
            raise TypeError(
                f"Unsupported column type for column '{col_name}': {col.dtype}"
            )
    return cpp_df


def _prepare_input_data(
    X: Union[pd.DataFrame, FloatMatrix], preprocess: bool
) -> Union[aplr_cpp.CppDataFrame, FloatMatrix]:
    """
    Prepares the input data for the C++ backend.

    If X is a pandas DataFrame, it's converted. If preprocess is True, it becomes
    a CppDataFrame. If preprocess is False, it's converted to a NumPy array.
    NumPy arrays are passed through as is.
    """
    if isinstance(X, pd.DataFrame):
        if preprocess:
            return _dataframe_to_cpp_dataframe(X)
        else:
            # Check if all columns are numeric before converting
            if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
                raise RuntimeError(
                    "Cannot convert DataFrame to matrix if it contains non-numeric columns. "
                    "Please ensure all columns are numeric or set preprocess=True."
                )
            return X.to_numpy(dtype=np.float64)
    return X


class APLRRegressor:
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
        preprocess: bool = True,
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
        self.preprocess = preprocess

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
        self.APLRRegressor.preprocess = self.preprocess

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
        self.__set_params_cpp()
        X = _prepare_input_data(X, self.preprocess)

        self.APLRRegressor.fit(
            X,
            y,
            sample_weight,
            X_names,
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
        X = _prepare_input_data(X, self.preprocess)

        if self.link_function == "custom_function":
            self.APLRRegressor.calculate_custom_transform_linear_predictor_to_predictions_function = (
                self.calculate_custom_transform_linear_predictor_to_predictions_function
            )
        return self.APLRRegressor.predict(X, cap_predictions_to_minmax_in_training)

    def set_term_names(self, X_names: List[str]):
        self.APLRRegressor.set_term_names(X_names)

    def calculate_feature_importance(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        sample_weight: FloatVector = np.empty(0),
    ) -> FloatVector:
        X = _prepare_input_data(X, self.preprocess)
        return self.APLRRegressor.calculate_feature_importance(X, sample_weight)

    def calculate_term_importance(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        sample_weight: FloatVector = np.empty(0),
    ) -> FloatVector:
        X = _prepare_input_data(X, self.preprocess)
        return self.APLRRegressor.calculate_term_importance(X, sample_weight)

    def calculate_local_feature_contribution(
        self, X: Union[pd.DataFrame, FloatMatrix]
    ) -> FloatMatrix:
        X = _prepare_input_data(X, self.preprocess)
        return self.APLRRegressor.calculate_local_feature_contribution(X)

    def calculate_local_term_contribution(
        self, X: Union[pd.DataFrame, FloatMatrix]
    ) -> FloatMatrix:
        X = _prepare_input_data(X, self.preprocess)
        return self.APLRRegressor.calculate_local_term_contribution(X)

    def calculate_local_contribution_from_selected_terms(
        self, X: Union[pd.DataFrame, FloatMatrix], predictor_indexes: List[int]
    ) -> FloatVector:
        X = _prepare_input_data(X, self.preprocess)
        return self.APLRRegressor.calculate_local_contribution_from_selected_terms(
            X, predictor_indexes
        )

    def calculate_terms(self, X: Union[pd.DataFrame, FloatMatrix]) -> FloatMatrix:
        X = _prepare_input_data(X, self.preprocess)
        return self.APLRRegressor.calculate_terms(X)

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
            "preprocess": self.preprocess,
        }

    # For sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__set_params_cpp()
        return self

    def __setstate__(self, state):
        # For backwards compatibility with older pickled models
        if "preprocess" not in state:
            state["preprocess"] = False
        self.__dict__.update(state)
        self.__set_params_cpp()


class APLRClassifier:
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
        preprocess: bool = True,
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
        self.preprocess = preprocess

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
        self.APLRClassifier.preprocess = self.preprocess

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
        self.__set_params_cpp()

        X = _prepare_input_data(X, self.preprocess)

        if isinstance(y, np.ndarray):
            y = y.astype(str).tolist()
        elif isinstance(y, list) and y and not isinstance(y[0], str):
            y = [str(val) for val in y]

        self.APLRClassifier.fit(
            X,
            y,
            sample_weight,
            X_names,
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
        X = _prepare_input_data(X, self.preprocess)

        return self.APLRClassifier.predict_class_probabilities(
            X, cap_predictions_to_minmax_in_training
        )

    def predict(
        self,
        X: Union[pd.DataFrame, FloatMatrix],
        cap_predictions_to_minmax_in_training: bool = False,
    ) -> List[str]:
        X = _prepare_input_data(X, self.preprocess)

        return self.APLRClassifier.predict(X, cap_predictions_to_minmax_in_training)

    def calculate_local_feature_contribution(
        self, X: Union[pd.DataFrame, FloatMatrix]
    ) -> FloatMatrix:
        X = _prepare_input_data(X, self.preprocess)
        return self.APLRClassifier.calculate_local_feature_contribution(X)

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
            preprocess=self.preprocess,
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
            "preprocess": self.preprocess,
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

    def __setstate__(self, state):
        # For backwards compatibility with older pickled models
        if "preprocess" not in state:
            state["preprocess"] = False
        self.__dict__.update(state)
        self.__set_params_cpp()


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
