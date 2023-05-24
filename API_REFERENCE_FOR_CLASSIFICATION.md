# APLRClassifier

## class aplr.APLRClassifier(m:int=9000, v:float=0.1, random_state:int=0, n_jobs:int=0, validation_ratio:float=0.2, bins:int=300, verbosity:int=0, max_interaction_level:int=1, max_interactions:int=100000, min_observations_in_split:int=20, ineligible_boosting_steps_added:int=10, max_eligible_terms:int=5)

### Constructor parameters

#### m (default = 9000)
The maximum number of boosting steps. If validation error does not flatten out at the end of the ***m***th boosting step, then try increasing it (or alternatively increase the learning rate).

#### v (default = 0.1)
The learning rate. Must be greater than zero and not more than one. The higher the faster the algorithm learns and the lower ***m*** is required. However, empirical evidence suggests that ***v <= 0.1*** gives better results. If the algorithm learns too fast (requires few boosting steps to converge) then try lowering the learning rate. Computational costs can be reduced by increasing the learning rate while simultaneously decreasing ***m***, potentially at the expense of predictiveness.

#### random_state (default = 0)
Used to randomly split training observations into training and validation if ***validation_set_indexes*** is not specified when fitting.

#### n_jobs (default = 0)
Multi-threading parameter. If ***0*** then uses all available cores for multi-threading. Any other positive integer specifies the number of cores to use (***1*** means single-threading).

#### validation_ratio (default = 0.2)
The ratio of training observations to use for validation instead of training. The number of boosting steps is automatically tuned to minimize validation error.

#### bins (default = 300)
Specifies the maximum number of bins to discretize the data into when searching for the best split. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs. Must be greater than 1.

#### verbosity (default = 0)
***0*** does not print progress reports during fitting. ***1*** prints a summary after running the ***fit*** method. ***2*** prints a summary after each boosting step.

#### max_interaction_level (default = 1)
Specifies the maximum allowed depth of interaction terms. ***0*** means that interactions are not allowed. This hyperparameter should be tuned.

#### max_interactions (default = 100000)
The maximum number of interactions allowed. A lower value may be used to reduce computational time.

#### min_observations_in_split (default = 20)
The minimum effective number of observations that a term in the model must rely on. This hyperparameter should be tuned. Larger values are more appropriate for larger datasets. Larger values result in more robust models (lower variance), potentially at the expense of increased bias.

#### ineligible_boosting_steps_added (default = 10)
Controls how many boosting steps a term that becomes ineligible has to remain ineligible. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs.

#### max_eligible_terms (default = 5)
Limits 1) the number of terms already in the model that can be considered as interaction partners in a boosting step and 2) how many terms remain eligible in the next boosting step. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs.


## Method: fit(X:npt.ArrayLike, y:List[str], sample_weight:npt.ArrayLike = np.empty(0), X_names:List[str]=[], validation_set_indexes:List[int]=[], prioritized_predictors_indexes:List[int]=[], monotonic_constraints:List[int]=[], interaction_constraints:List[int]=[])

***This method fits the model to data.***

### Parameters

#### X
A numpy matrix with predictor values.

#### y
A numpy vector with response values.

#### sample_weight
An optional numpy vector with sample weights. If not specified then the observations are weighted equally.

#### X_names
An optional list of strings containing names for each predictor in ***X***. Naming predictors may increase model readability because model terms get names based on ***X_names***.

#### validation_set_indexes
An optional list of integers specifying the indexes of observations to be used for validation instead of training. If this is specified then ***validation_ratio*** is not used. Specifying ***validation_set_indexes*** may be useful for example when modelling time series data (you can place more recent observations in the validation set).

#### prioritized_predictors_indexes
An optional list of integers specifying the indexes of predictors (columns) in ***X*** that should be prioritized. Terms of the prioritized predictors will enter the model as long as they reduce the training error and do not contain too few effective observations. They will also be updated more often.

#### monotonic_constraints
An optional list of integers specifying monotonic constraints on model terms. For example, if there are three predictors in ***X***, then monotonic_constraints = [1,0,-1] means that 1) the first predictor in ***X*** cannot be used in interaction terms as a secondary effect and all terms using the first predictor in ***X*** as a main effect must have positive regression coefficients, 2) there are no monotonic constraints on terms using the second predictor in ***X***, and 3) the third predictor in ***X*** cannot be used in interaction terms as a secondary effect and all terms using the third predictor in ***X*** as a main effect must have negative regression coefficients.

#### interaction_constraints
An optional list of integers specifying interaction constraints on model terms. For example, if there are three predictors in ***X***, then interaction_constraints = [1,0,2] means that 1) the first predictor in ***X*** cannot be used in interaction terms as a secondary effect, 2) there are no interaction constraints on terms using the second predictor in ***X***, and 3) the third predictor in ***X*** cannot be used in any interaction terms.


## Method: predict_class_probabilities(X:npt.ArrayLike, cap_predictions_to_minmax_in_training:bool=False)

***Returns a numpy matrix containing predictions of the data in X. Requires that the model has been fitted with the fit method.***

### Parameters

#### X
A numpy matrix with predictor values.

#### cap_predictions_to_minmax_in_training
If ***True*** then for each underlying logit model the predictions are capped so that they are not less than the minimum and not greater than the maximum prediction or response in the training dataset.


## Method: predict(X:npt.ArrayLike, cap_predictions_to_minmax_in_training:bool=False)

***Returns a list of strings containing predictions of the data in X. An observation is classified to the category with the highest predicted class probability. Requires that the model has been fitted with the fit method.***

### Parameters
Parameters are the same as in ***predict_class_probabilities()***.


## Method: calculate_local_feature_importance(X:npt.ArrayLike)

***Returns a numpy matrix containing local feature importance for new data by each predictor in X.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: get_categories()

***Returns a list containing the names of each category.***


## Method: get_logit_model(category:str)

***Returns the logit model (of type APLRRegressor) that predicts whether an observation belongs to class ***category*** or not. The logit model can be used for example to inspect which terms are in the model.***

### Parameters

#### category
A string specifying the label of the category.


## Method: get_validation_indexes()

***Returns a list of integers containing the indexes of the training data observations used for validation and not training.***


## Method: get_validation_error_steps()

***Returns a numpy vector containing the validation error by boosting step (average of log loss for each underlying logit model). Use this to determine if the maximum number of boosting steps (m) or learning rate (v) should be changed.***


## Method: get_validation_error()

***Returns the validation error measured by the average of log loss for each underlying logit model.***


## Method: get_feature_importance()

***Returns a numpy vector containing the feature importance of each predictor, estimated on the validation set as an average of feature importances for the underlying logit models.***