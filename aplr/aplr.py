import numpy as np
import numpy.typing as npt
from typing import List
import aplr_cpp


class APLRRegressor():
    def __init__(self, m:int=1000, v:float=0.1, random_state:int=0, family:str="gaussian", link_function:str="identity", n_jobs:int=0, validation_ratio:float=0.2, intercept:float=np.nan, bins:int=300, max_interaction_level:int=1, max_interactions:int=100000, min_observations_in_split:int=20, ineligible_boosting_steps_added:int=10, max_eligible_terms:int=5, verbosity:int=0, tweedie_power:float=1.5):
        self.m=m
        self.v=v
        self.random_state=random_state
        self.family=family
        self.link_function=link_function
        self.n_jobs=n_jobs
        self.validation_ratio=validation_ratio
        self.intercept=intercept
        self.bins=bins
        self.max_interaction_level=max_interaction_level
        self.max_interactions=max_interactions
        self.min_observations_in_split=min_observations_in_split
        self.ineligible_boosting_steps_added=ineligible_boosting_steps_added
        self.max_eligible_terms=max_eligible_terms
        self.verbosity=verbosity
        self.tweedie_power=tweedie_power

        #Creating aplr_cpp and setting parameters
        self.APLRRegressor=aplr_cpp.APLRRegressor()
        self.__set_params_cpp()

    #Sets parameters for aplr_cpp.APLRRegressor cpp object
    def __set_params_cpp(self):
        self.APLRRegressor.m=self.m
        self.APLRRegressor.v=self.v
        self.APLRRegressor.random_state=self.random_state
        self.APLRRegressor.family=self.family
        self.APLRRegressor.link_function=self.link_function
        self.APLRRegressor.n_jobs=self.n_jobs
        self.APLRRegressor.validation_ratio=self.validation_ratio
        self.APLRRegressor.intercept=self.intercept
        self.APLRRegressor.bins=self.bins
        self.APLRRegressor.max_interaction_level=self.max_interaction_level
        self.APLRRegressor.max_interactions=self.max_interactions
        self.APLRRegressor.min_observations_in_split=self.min_observations_in_split
        self.APLRRegressor.ineligible_boosting_steps_added=self.ineligible_boosting_steps_added
        self.APLRRegressor.max_eligible_terms=self.max_eligible_terms
        self.APLRRegressor.verbosity=self.verbosity
        self.APLRRegressor.tweedie_power=self.tweedie_power

    def fit(self, X:npt.ArrayLike, y:npt.ArrayLike, sample_weight:npt.ArrayLike = np.empty(0), X_names:List[str]=[], validation_set_indexes:List[int]=[]):
        self.__set_params_cpp()
        self.APLRRegressor.fit(X,y,sample_weight,X_names,validation_set_indexes)

    def predict(self, X:npt.ArrayLike, cap_predictions_to_minmax_in_training:bool=True)->npt.ArrayLike:
        return self.APLRRegressor.predict(X, cap_predictions_to_minmax_in_training)

    def set_term_names(self, X_names:List[str]):
        self.APLRRegressor.set_term_names(X_names)

    def calculate_local_feature_importance(self,X:npt.ArrayLike)->npt.ArrayLike:
        return self.APLRRegressor.calculate_local_feature_importance(X)

    def calculate_local_feature_importance_for_terms(self,X:npt.ArrayLike)->npt.ArrayLike:
        return self.APLRRegressor.calculate_local_feature_importance_for_terms(X)

    def calculate_terms(self,X:npt.ArrayLike)->npt.ArrayLike:
        return self.APLRRegressor.calculate_terms(X)

    def get_term_names(self)->List[str]:
        return self.APLRRegressor.get_term_names()

    def get_term_coefficients(self)->npt.ArrayLike:
        return self.APLRRegressor.get_term_coefficients()

    def get_term_coefficient_steps(self, term_index:int)->npt.ArrayLike:
        return self.APLRRegressor.get_term_coefficient_steps(term_index)
    
    def get_validation_error_steps(self)->npt.ArrayLike:
        return self.APLRRegressor.get_validation_error_steps()

    def get_feature_importance(self)->npt.ArrayLike:
        return self.APLRRegressor.get_feature_importance()

    def get_intercept(self)->float:
        return self.APLRRegressor.get_intercept()

    def get_intercept_steps(self)->npt.ArrayLike:
        return self.APLRRegressor.get_intercept_steps()

    def get_m(self)->int:
        return self.APLRRegressor.get_m()

    #For sklearn
    def get_params(self, deep=True):
        return {"m": self.m, "v": self.v,"random_state":self.random_state,"family":self.family,"link_function":self.link_function,"n_jobs":self.n_jobs,"validation_ratio":self.validation_ratio,"intercept":self.intercept,"bins":self.bins,"max_interaction_level":self.max_interaction_level,"max_interactions":self.max_interactions,"verbosity":self.verbosity,"min_observations_in_split":self.min_observations_in_split,"ineligible_boosting_steps_added":self.ineligible_boosting_steps_added,"max_eligible_terms":self.max_eligible_terms,"tweedie_power":self.tweedie_power}

    #For sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__set_params_cpp()
        return self