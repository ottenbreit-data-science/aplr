# APLRTuner

## class aplr.APLRTuner(parameters: Union[Dict[str, List[float]], List[Dict[str, List[float]]]] = {"max_interaction_level": [0, 1], "min_observations_in_split": [4, 10, 20, 100, 500, 1000]}, is_regressor: bool = True)

### Constructor parameters

#### parameters (default = {"max_interaction_level": [0, 1], "min_observations_in_split": [4, 10, 20, 100, 500, 1000]})
The parameters that you wish to tune.

#### is_regressor (default = True)
Whether you want to use APLRRegressor (True) or APLRClassifier (False).


## Method: fit(X: FloatMatrix, y: FloatVector, **kwargs)

***This method tunes the model to data.***

### Parameters

#### X
A numpy matrix with predictor values.

#### y
A numpy vector with response values.

#### kwargs
Optional parameters sent to the fit methods in the underlying APLRRegressor or APLRClassifier models.


## Method: predict(X: FloatMatrix, **kwargs)

***Returns the predictions of the best tuned model as a numpy array if regression or as a list of strings if classification.***

### Parameters

#### X
A numpy matrix with predictor values.

#### kwargs
Optional parameters sent to the predict method in the best tuned model.


## Method: predict_class_probabilities(X: FloatMatrix, **kwargs)

***This method returns predicted class probabilities of the best tuned model as a numpy matrix.***

### Parameters

#### X
A numpy matrix with predictor values.

#### kwargs
Optional parameters sent to the predict_class_probabilities method in the best tuned model.


## Method: predict_proba(X: FloatMatrix, **kwargs)

***This method returns predicted class probabilities of the best tuned model as a numpy matrix. Similar to the predict_class_probabilities method but the name predict_proba is compatible with scikit-learn.***

### Parameters

#### X
A numpy matrix with predictor values.

#### kwargs
Optional parameters sent to the predict_class_probabilities method in the best tuned model.


## Method: get_best_estimator()

***Returns the best tuned model. This is an APLRRegressor or APLRClassifier object.***


## Method: get_cv_results()

***Returns the cv results from the tuning as a list of dictionaries, List[Dict[str, float]].***