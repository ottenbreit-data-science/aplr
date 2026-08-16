# APLRTuner

## class aplr.APLRTuner(parameters: Dict[str, List[Any]] = {"m": [3000], "v": [0.5], "max_interaction_level": [1], "penalty_for_interactions": [0.0, 0.5, 1.0], "min_observations_in_split": [0.3, 0.4, 0.5], "ridge_penalty": [0, 0.0001]}, is_regressor: bool = True, sequential_tuning: bool = False)

### Constructor parameters

#### parameters (default = {"m": [3000], "v": [0.5], "max_interaction_level": [1], "penalty_for_interactions": [0.0, 0.5, 1.0], "min_observations_in_split": [0.3, 0.4, 0.5], "ridge_penalty": [0, 0.0001]})
A dictionary where keys are parameter names and values are lists of parameter settings to try.

#### is_regressor (default = True)
Whether you want to use APLRRegressor (True) or APLRClassifier (False).

#### sequential_tuning (default = False)
If True, hyperparameters are tuned sequentially instead of performing a full grid search. The tuning order is determined by the key order in the `parameters` dictionary. This can be much faster but may not find the global optimum.


## Method: fit(X: Union[pd.DataFrame, FloatMatrix], y: FloatVector, **kwargs)

***This method tunes the model to data.***

### Parameters

#### X
A numpy matrix or pandas DataFrame with predictor values.

#### y
A numpy vector with response values.

#### kwargs
Optional parameters sent to the fit methods in the underlying APLRRegressor or APLRClassifier models.


## Method: predict(X: Union[pd.DataFrame, FloatMatrix], **kwargs)

***Returns the predictions of the best tuned model as a numpy array if regression or as a list of strings if classification.***

### Parameters

#### X
A numpy matrix or pandas DataFrame with predictor values.

#### kwargs
Optional parameters sent to the predict method in the best tuned model.


## Method: predict_class_probabilities(X: Union[pd.DataFrame, FloatMatrix], **kwargs)

***This method returns predicted class probabilities of the best tuned model as a numpy matrix.***

### Parameters

#### X
A numpy matrix or pandas DataFrame with predictor values.

#### kwargs
Optional parameters sent to the predict_class_probabilities method in the best tuned model.


## Method: predict_proba(X: Union[pd.DataFrame, FloatMatrix], **kwargs)

***This method returns predicted class probabilities of the best tuned model as a numpy matrix. Similar to the predict_class_probabilities method but the name predict_proba is compatible with scikit-learn.***

### Parameters

#### X
A numpy matrix or pandas DataFrame with predictor values.

#### kwargs
Optional parameters sent to the predict_class_probabilities method in the best tuned model.


## Method: get_best_estimator()

***Returns the best tuned model. This is an APLRRegressor or APLRClassifier object.***


## Method: get_cv_results()

***Returns the cv results from the tuning as a list of dictionaries, List[Dict[str, Any]].***