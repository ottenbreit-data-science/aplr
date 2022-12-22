# APLRRegressor

## class aplr.APLRRegressor(m:int=1000, v:float=0.1, random_state:int=0, family:str="gaussian", link_function:str="identity", n_jobs:int=0, validation_ratio:float=0.2, intercept:float=np.nan, bins:int=300, max_interaction_level:int=1, max_interactions:int=100000, min_observations_in_split:int=20, ineligible_boosting_steps_added:int=10, max_eligible_terms:int=5, verbosity:int=0, tweedie_power:float=1.5, group_size_for_validation_group_mse:int=100)

### Constructor parameters

#### m (default = 1000)
The maximum number of boosting steps. If validation error does not flatten out at the end of the ***m***th boosting step, then try increasing it (or alternatively increase the learning rate).

#### v (default = 0.1)
The learning rate. Must be greater than zero and not more than one. The higher the faster the algorithm learns and the lower ***m*** is required. However, empirical evidence suggests that ***v <= 0.1*** gives better results. If the algorithm learns too fast (requires few boosting steps to converge) then try lowering the learning rate. Computational costs can be reduced by increasing the learning rate while simultaneously decreasing ***m***, potentially at the expense of predictiveness.

#### random_state (default = 0)
Used to randomly split training observations into training and validation if ***validation_set_indexes*** is not specified when fitting.

#### family (default = "gaussian")
Determines the loss function used. Allowed values are "gaussian", "binomial", "poisson", "gamma" and "tweedie". This is used together with ***link_function***. ***family*** is not intended to be a tuning parameter because it defines how the loss function is calculated. However, if you wish to tune it then the method ***get_validation_group_mse()*** provides a useful tuning metric.

#### link_function (default = "identity")
Determines how the linear predictor is transformed to predictions. Allowed values are "identity", "logit" and "log". For an ordinary regression model use ***family*** "gaussian" and ***link_function*** "identity". For logistic regression use ***family*** "binomial" and ***link_function*** "logit". For a multiplicative model use the "log" ***link_function***. The "log" ***link_function*** often works best with a "poisson", "gamma" or "tweedie" ***family***, depending on the data. The ***family*** "poisson", "gamma" or "tweedie" should only be used with the "log" ***link_function***. Inappropriate combinations of ***family*** and ***link_function*** may result in a warning message when fitting the model and/or a poor model fit. Please note that values other than "identity" typically require a significantly higher ***m*** (or ***v***) in order to converge. ***link_function*** is not intended to be a tuning parameter because it defines the model structure. However, if you wish to tune it then the method ***get_validation_group_mse()*** provides a useful tuning metric.

#### n_jobs (default = 0)
Multi-threading parameter. If ***0*** then uses all available cores for multi-threading. Any other positive integer specifies the number of cores to use (***1*** means single-threading).

#### validation_ratio (default = 0.2)
The ratio of training observations to use for validation instead of training. The number of boosting steps is automatically tuned to minimize validation error.

#### intercept (default = nan)
Specifies the intercept term of the model if you want to predict before doing any training. However, when the ***fit*** method is run then the intercept is estimated based on data and whatever was specified as ***intercept*** when instantiating ***APLRRegressor*** gets overwritten.

#### bins (default = 300)
Specifies the maximum number of bins to discretize the data into when searching for the best split. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs.

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

#### verbosity (default = 0)
***0*** does not print progress reports during fitting. ***1*** prints a summary after running the ***fit*** method. ***2*** prints a summary after each boosting step.

#### tweedie_power (default = 1.5)
Species the variance power for the "tweedie" ***family***. It can be useful to tune this hyperparameter. The method ***get_validation_group_mse()*** provides a tuning metric that ***tweedie_power*** can be tuned on.

#### group_size_for_validation_group_mse (default = 100)
APLR calculates mean squared error on grouped data in the validation set. This can be useful for comparing models that have different ***family*** or ***tweedie_power*** parameters. The maximum number of observations in each group is specified by  ***group_size_for_validation_group_mse***. Some of the observations with the lowest or highest response values will belong to groups with less than   ***group_size_for_validation_group_mse*** observations. The minimum number of observations in a group is ***group_size_for_validation_group_mse/2***. If ***group_size_for_validation_group_mse*** is equal to or higher than the number of observations in the validation set, then there will only be one group (in this case the grouped validation MSE is not so useful). ***group_size_for_validation_group_mse*** should be large enough so that the Central Limit Theorem holds (at least 60, but 100 is a safer choice). Also, the number of observations in the validation set should be substantially higher than ***group_size_for_validation_group_mse*** for group validation MSE to be useful.


## Method: fit(X:npt.ArrayLike, y:npt.ArrayLike, sample_weight:npt.ArrayLike = np.empty(0), X_names:List[str]=[], validation_set_indexes:List[int]=[])

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


## Method: predict(X:npt.ArrayLike, cap_predictions_to_minmax_in_training:bool=True)

***Returns a numpy vector containing predictions of the data in X. Requires that the model has been fitted with the fit method.***

### Parameters

#### X
A numpy matrix with predictor values.

#### cap_predictions_to_minmax_in_training
If ***True*** then predictions are capped so that they are not less than the minimum and not greater than the maximum prediction in the training dataset. This is recommended especially if ***max_interaction_level*** is high. However, if you need the model to extrapolate then set this parameter to ***False***.


## Method: set_term_names(X_names:List[str])

***This method sets the names of terms based on X_names.***

### Parameters

#### X_names
A list of strings containing names for each predictor in the ***X*** matrix that the model was trained on.


## Method: calculate_local_feature_importance(X:npt.ArrayLike)

***Returns a numpy matrix containing local feature importance for new data by each predictor in X.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: calculate_local_feature_importance_for_terms(X:npt.ArrayLike)

***Returns a numpy matrix containing local feature importance for new data by each term in the model.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: calculate_terms(X:npt.ArrayLike)

***Returns a numpy matrix containing values of model terms calculated on X.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: get_term_names()

***Returns a list of strings containing term names.***


## Method: get_term_coefficients()

***Returns a numpy vector containing term regression coefficients.***


## Method: get_term_coefficient_steps(term_index:int)

***Returns a numpy vector containing regression coefficients by each boosting step for the term selected.***

### Parameters

#### term_index
The index of the term selected. So ***0*** is the first term, ***1*** is the second term and so on.


## Method: get_validation_error_steps()

***Returns a numpy vector containing the validation error by boosting step. Use this to determine if the maximum number of boosting steps (m) or learning rate (v) should be changed.***


## Method: get_feature_importance()

***Returns a numpy vector containing the feature importance (estimated on the validation set) of each predictor.***


## Method: get_intercept()

***Returns the regression coefficient of the intercept term.***


## Method: get_intercept_steps()

***Returns a numpy vector containing the regression coefficients of the intercept term by boosting step.***


## Method: get_m()

***Returns the number of boosting steps in the model (the value that minimized validation error).***


## Method: get_validation_group_mse()

***Returns mean squared error on grouped data in the validation set.*** See ***group_size_for_validation_group_mse*** for more information.