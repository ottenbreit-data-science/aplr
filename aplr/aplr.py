import numpy as np
import numpy.typing as npt
from typing import List, Callable, Optional
import aplr_cpp


class APLRRegressor:
    def __init__(
        self,
        m: int = 1000,
        v: float = 0.1,
        random_state: int = 0,
        loss_function: str = "mse",
        link_function: str = "identity",
        n_jobs: int = 0,
        validation_ratio: float = 0.2,
        bins: int = 300,
        max_interaction_level: int = 1,
        max_interactions: int = 100000,
        min_observations_in_split: int = 20,
        ineligible_boosting_steps_added: int = 10,
        max_eligible_terms: int = 5,
        verbosity: int = 0,
        dispersion_parameter: float = 1.5,
        validation_tuning_metric: str = "default",
        quantile: float = 0.5,
        calculate_custom_validation_error_function: Optional[
            Callable[
                [npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], float
            ]
        ] = None,
        calculate_custom_loss_function: Optional[
            Callable[
                [npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], float
            ]
        ] = None,
        calculate_custom_negative_gradient_function: Optional[
            Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]
        ] = None,
        calculate_custom_transform_linear_predictor_to_predictions_function: Optional[
            Callable[[npt.ArrayLike], npt.ArrayLike]
        ] = None,
        calculate_custom_differentiate_predictions_wrt_linear_predictor_function: Optional[
            Callable[[npt.ArrayLike], npt.ArrayLike]
        ] = None,
    ):
        self.m = m
        self.v = v
        self.random_state = random_state
        self.loss_function = loss_function
        self.link_function = link_function
        self.n_jobs = n_jobs
        self.validation_ratio = validation_ratio
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
        self.APLRRegressor.validation_ratio = self.validation_ratio
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

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike = np.empty(0),
        X_names: List[str] = [],
        validation_set_indexes: List[int] = [],
        prioritized_predictors_indexes: List[int] = [],
        monotonic_constraints: List[int] = [],
        group: npt.ArrayLike = np.empty(0),
        interaction_constraints: List[int] = [],
    ):
        self.__set_params_cpp()
        self.APLRRegressor.fit(
            X,
            y,
            sample_weight,
            X_names,
            validation_set_indexes,
            prioritized_predictors_indexes,
            monotonic_constraints,
            group,
            interaction_constraints,
        )

    def predict(
        self, X: npt.ArrayLike, cap_predictions_to_minmax_in_training: bool = True
    ) -> npt.ArrayLike:
        if self.link_function == "custom_function":
            self.APLRRegressor.calculate_custom_transform_linear_predictor_to_predictions_function = (
                self.calculate_custom_transform_linear_predictor_to_predictions_function
            )
        return self.APLRRegressor.predict(X, cap_predictions_to_minmax_in_training)

    def set_term_names(self, X_names: List[str]):
        self.APLRRegressor.set_term_names(X_names)

    def calculate_local_feature_importance(self, X: npt.ArrayLike) -> npt.ArrayLike:
        return self.APLRRegressor.calculate_local_feature_importance(X)

    def calculate_local_feature_importance_for_terms(
        self, X: npt.ArrayLike
    ) -> npt.ArrayLike:
        return self.APLRRegressor.calculate_local_feature_importance_for_terms(X)

    def calculate_terms(self, X: npt.ArrayLike) -> npt.ArrayLike:
        return self.APLRRegressor.calculate_terms(X)

    def get_term_names(self) -> List[str]:
        return self.APLRRegressor.get_term_names()

    def get_term_coefficients(self) -> npt.ArrayLike:
        return self.APLRRegressor.get_term_coefficients()

    def get_term_coefficient_steps(self, term_index: int) -> npt.ArrayLike:
        return self.APLRRegressor.get_term_coefficient_steps(term_index)

    def get_validation_error_steps(self) -> npt.ArrayLike:
        return self.APLRRegressor.get_validation_error_steps()

    def get_feature_importance(self) -> npt.ArrayLike:
        return self.APLRRegressor.get_feature_importance()

    def get_intercept(self) -> float:
        return self.APLRRegressor.get_intercept()

    def get_intercept_steps(self) -> npt.ArrayLike:
        return self.APLRRegressor.get_intercept_steps()

    def get_optimal_m(self) -> int:
        return self.APLRRegressor.get_optimal_m()

    def get_validation_tuning_metric(self) -> str:
        return self.APLRRegressor.get_validation_tuning_metric()

    def get_validation_indexes(self) -> List[int]:
        return self.APLRRegressor.get_validation_indexes()

    # For sklearn
    def get_params(self, deep=True):
        return {
            "m": self.m,
            "v": self.v,
            "random_state": self.random_state,
            "loss_function": self.loss_function,
            "link_function": self.link_function,
            "n_jobs": self.n_jobs,
            "validation_ratio": self.validation_ratio,
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
        }

    # For sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__set_params_cpp()
        return self


class APLRClassifier:
    def __init__(
        self,
        m: int = 9000,
        v: float = 0.1,
        random_state: int = 0,
        n_jobs: int = 0,
        validation_ratio: float = 0.2,
        bins: int = 300,
        verbosity: int = 0,
        max_interaction_level: int = 1,
        max_interactions: int = 100000,
        min_observations_in_split: int = 20,
        ineligible_boosting_steps_added: int = 10,
        max_eligible_terms: int = 5,
    ):
        self.m = m
        self.v = v
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.validation_ratio = validation_ratio
        self.bins = bins
        self.verbosity = verbosity
        self.max_interaction_level = max_interaction_level
        self.max_interactions = max_interactions
        self.min_observations_in_split = min_observations_in_split
        self.ineligible_boosting_steps_added = ineligible_boosting_steps_added
        self.max_eligible_terms = max_eligible_terms

        # Creating aplr_cpp and setting parameters
        self.APLRClassifier = aplr_cpp.APLRClassifier()
        self.__set_params_cpp()

    # Sets parameters for aplr_cpp.APLRClassifier cpp object
    def __set_params_cpp(self):
        self.APLRClassifier.m = self.m
        self.APLRClassifier.v = self.v
        self.APLRClassifier.random_state = self.random_state
        self.APLRClassifier.n_jobs = self.n_jobs
        self.APLRClassifier.validation_ratio = self.validation_ratio
        self.APLRClassifier.bins = self.bins
        self.APLRClassifier.verbosity = self.verbosity
        self.APLRClassifier.max_interaction_level = self.max_interaction_level
        self.APLRClassifier.max_interactions = self.max_interactions
        self.APLRClassifier.min_observations_in_split = self.min_observations_in_split
        self.APLRClassifier.ineligible_boosting_steps_added = (
            self.ineligible_boosting_steps_added
        )
        self.APLRClassifier.max_eligible_terms = self.max_eligible_terms

    def fit(
        self,
        X: npt.ArrayLike,
        y: List[str],
        sample_weight: npt.ArrayLike = np.empty(0),
        X_names: List[str] = [],
        validation_set_indexes: List[int] = [],
        prioritized_predictors_indexes: List[int] = [],
        monotonic_constraints: List[int] = [],
        interaction_constraints: List[int] = [],
    ):
        self.__set_params_cpp()
        self.APLRClassifier.fit(
            X,
            y,
            sample_weight,
            X_names,
            validation_set_indexes,
            prioritized_predictors_indexes,
            monotonic_constraints,
            interaction_constraints,
        )

    def predict_class_probabilities(
        self, X: npt.ArrayLike, cap_predictions_to_minmax_in_training: bool = False
    ) -> npt.ArrayLike:
        return self.APLRClassifier.predict_class_probabilities(
            X, cap_predictions_to_minmax_in_training
        )

    def predict(
        self, X: npt.ArrayLike, cap_predictions_to_minmax_in_training: bool = False
    ) -> List[str]:
        return self.APLRClassifier.predict(X, cap_predictions_to_minmax_in_training)

    def calculate_local_feature_importance(self, X: npt.ArrayLike) -> npt.ArrayLike:
        return self.APLRClassifier.calculate_local_feature_importance(X)

    def get_categories(self) -> List[str]:
        return self.APLRClassifier.get_categories()

    def get_logit_model(self, category: str) -> APLRRegressor:
        return self.APLRClassifier.get_logit_model(category)

    def get_validation_indexes(self) -> List[int]:
        return self.APLRClassifier.get_validation_indexes()

    def get_validation_error_steps(self) -> npt.ArrayLike:
        return self.APLRClassifier.get_validation_error_steps()

    def get_validation_error(self) -> float:
        return self.APLRClassifier.get_validation_error()

    def get_feature_importance(self) -> npt.ArrayLike:
        return self.APLRClassifier.get_feature_importance()

    # For sklearn
    def get_params(self, deep=True):
        return {
            "m": self.m,
            "v": self.v,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "validation_ratio": self.validation_ratio,
            "bins": self.bins,
            "verbosity": self.verbosity,
            "max_interaction_level": self.max_interaction_level,
            "max_interactions": self.max_interactions,
            "min_observations_in_split": self.min_observations_in_split,
            "ineligible_boosting_steps_added": self.ineligible_boosting_steps_added,
            "max_eligible_terms": self.max_eligible_terms,
        }

    # For sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__set_params_cpp()
        return self
