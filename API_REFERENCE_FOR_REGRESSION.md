# APLRRegressor

## class aplr.APLRRegressor(m:int=1000, v:float=0.1, random_state:int=0, loss_function:str="mse", link_function:str="identity", n_jobs:int=0, validation_ratio:float=0.2, bins:int=100, max_interaction_level:int=1, max_interactions:int=100000, min_observations_in_split:int=20, ineligible_boosting_steps_added:int=20, max_eligible_terms:int=10, verbosity:int=0, dispersion_parameter:float=1.5, validation_tuning_metric:str="default", quantile:float=0.5, calculate_custom_validation_error_function:Optional[Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], float]]=None, calculate_custom_loss_function:Optional[Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], float]]=None, calculate_custom_negative_gradient_function:Optional[Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]]=None, calculate_custom_transform_linear_predictor_to_predictions_function:Optional[Callable[[npt.ArrayLike], npt.ArrayLike]]=None, calculate_custom_differentiate_predictions_wrt_linear_predictor_function:Optional[Callable[[npt.ArrayLike], npt.ArrayLike]]=None, boosting_steps_before_pruning_is_done: int = 0, boosting_steps_before_interactions_are_allowed: int = 0)

### Constructor parameters

#### m (default = 1000)
The maximum number of boosting steps. If validation error does not flatten out at the end of the ***m***th boosting step, then try increasing it (or alternatively increase the learning rate).

#### v (default = 0.1)
The learning rate. Must be greater than zero and not more than one. The higher the faster the algorithm learns and the lower ***m*** is required. However, empirical evidence suggests that ***v <= 0.1*** gives better results. If the algorithm learns too fast (requires few boosting steps to converge) then try lowering the learning rate. Computational costs can be reduced by increasing the learning rate while simultaneously decreasing ***m***, potentially at the expense of predictiveness.

#### random_state (default = 0)
Used to randomly split training observations into training and validation if ***validation_set_indexes*** is not specified when fitting.

#### loss_function (default = "mse")
Determines the loss function used. Allowed values are "mse", "binomial", "poisson", "gamma", "tweedie", "group_mse", "mae", "quantile", "negative_binomial", "cauchy", "weibull" and "custom_function". This is used together with ***link_function***. When ***loss_function*** is "group_mse" then the "group" argument in the ***fit*** method must be provided. In the latter case APLR will try to minimize group MSE when training the model. The ***loss_function*** "quantile" is used together with the ***quantile*** constructor parameter. When ***loss_function*** is "custom_function" then the constructor parameters ***calculate_custom_loss_function*** and ***calculate_custom_negative_gradient_function***, both described below, must be provided.

#### link_function (default = "identity")
Determines how the linear predictor is transformed to predictions. Allowed values are "identity", "logit", "log" and "custom_function". For an ordinary regression model use ***loss_function*** "mse" and ***link_function*** "identity". For logistic regression use ***loss_function*** "binomial" and ***link_function*** "logit". For a multiplicative model use the "log" ***link_function***. The "log" ***link_function*** often works best with a "poisson", "gamma", "tweedie", "negative_binomial" or "weibull" ***loss_function***, depending on the data. The ***loss_function*** "poisson", "gamma", "tweedie", "negative_binomial" or "weibull" should only be used with the "log" ***link_function***. Inappropriate combinations of ***loss_function*** and ***link_function*** may result in a warning message when fitting the model and/or a poor model fit. Please note that values other than "identity" typically require a significantly higher ***m*** (or ***v***) in order to converge. When ***link_function*** is "custom_function" then the constructor parameters ***calculate_custom_transform_linear_predictor_to_predictions_function*** and ***calculate_custom_differentiate_predictions_wrt_linear_predictor_function***, both described below, must be provided.

#### n_jobs (default = 0)
Multi-threading parameter. If ***0*** then uses all available cores for multi-threading. Any other positive integer specifies the number of cores to use (***1*** means single-threading).

#### validation_ratio (default = 0.2)
The ratio of training observations to use for validation instead of training. The number of boosting steps is automatically tuned to minimize validation error.

#### bins (default = 100)
Specifies the maximum number of bins to discretize the data into when searching for the best split. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs. Must be greater than 1.

#### max_interaction_level (default = 1)
Specifies the maximum allowed depth of interaction terms. ***0*** means that interactions are not allowed. This hyperparameter should be tuned.

#### max_interactions (default = 100000)
The maximum number of interactions allowed. A lower value may be used to reduce computational time.

#### min_observations_in_split (default = 20)
The minimum effective number of observations that a term in the model must rely on. This hyperparameter should be tuned. Larger values are more appropriate for larger datasets. Larger values result in more robust models (lower variance), potentially at the expense of increased bias.

#### ineligible_boosting_steps_added (default = 20)
Controls how many boosting steps a term that becomes ineligible has to remain ineligible. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs.

#### max_eligible_terms (default = 10)
Limits 1) the number of terms already in the model that can be considered as interaction partners in a boosting step and 2) how many terms remain eligible in the next boosting step. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs.

#### verbosity (default = 0)
***0*** does not print progress reports during fitting. ***1*** prints a summary after running the ***fit*** method. ***2*** prints a summary after each boosting step.

#### dispersion_parameter (default = 1.5)
Specifies the variance power when ***loss_function*** is "tweedie". Specifies a dispersion parameter when ***loss_function*** is "negative_binomial", "cauchy" or "weibull". 

#### validation_tuning_metric (default = "default")
Specifies which metric to use for validating the model and tuning ***m***. Available options are "default" (using the same methodology as when calculating the training error), "mse", "mae", "negative_gini", "rankability", "group_mse" and "custom_function". The default is often a choice that fits well with respect to the ***loss_function*** chosen. However, if you want to use ***loss_function*** or ***dispersion_parameter*** as tuning parameters then the default is not suitable. "rankability" uses a methodology similar to the one described in https://towardsdatascience.com/how-to-calculate-roc-auc-score-for-regression-models-c0be4fdf76bb except that the metric is inverted and can be weighted by sample weights. "group_mse" requires that the "group" argument in the ***fit*** method is provided. For "custom_function" see ***calculate_custom_validation_error_function*** below.

#### quantile (default = 0.5)
Specifies the quantile to use when ***loss_function*** is "quantile".

#### calculate_custom_validation_error_function (default = None)
A Python function that calculates validation error if ***validation_tuning_metric*** is "custom_function". Example:

```
def custom_validation_error_function(y, predictions, sample_weight, group, other_data):
    squared_errors = (y-predictions)**2
    return squared_errors.mean()
```

#### calculate_custom_loss_function (default = None)
A Python function that calculates loss if ***loss_function*** is "custom_function". Example:

```
def custom_loss_function(y, predictions, sample_weight, group, other_data):
    squared_errors = (y-predictions)**2
    return squared_errors.mean()
```

#### calculate_custom_negative_gradient_function (default = None)
A Python function that calculates the negative gradient if ***loss_function*** is "custom_function". The negative gradient should be proportional to the negative of the first order differentiation of the custom loss function (***calculate_custom_loss_function***) with respect to the predictions. Example:

```
def custom_negative_gradient_function(y, predictions, group, other_data):
    residuals = y-predictions
    return residuals
```

#### calculate_custom_transform_linear_predictor_to_predictions_function (default = None)
A Python function that transforms the linear predictor to predictions if ***link_function*** is "custom_function". Example:

```
def calculate_custom_transform_linear_predictor_to_predictions(linear_predictor):
    #This particular example is prone to numerical problems (requires small and non-negative response values).
    predictions=np.exp(linear_predictor)
    return predictions
```

#### calculate_custom_differentiate_predictions_wrt_linear_predictor_function (default = None)
A Python function that does a first order differentiation of the predictions with respect to the linear predictor. Example:

```
def calculate_custom_differentiate_predictions_wrt_linear_predictor(linear_predictor):
    #This particular example is prone to numerical problems (requires small and non-negative response values).
    differentiated_predictions=np.exp(linear_predictor)
    return differentiated_predictions
```

#### boosting_steps_before_pruning_is_done (default = 0)
Specifies how many boosting steps to wait before pruning the model. If 0 (default) then pruning is not done. If for example 500 then the model will be pruned in boosting steps 500, 1000, and so on. When pruning, terms are removed as long as this reduces the training error. This can be a computationally costly operation especially if the model gets many terms. Pruning may slightly improve predictiveness.

#### boosting_steps_before_interactions_are_allowed (default = 0)
Specifies how many boosting steps to wait before searching for interactions. If for example 800, then the algorithm will be forced to only fit main effects in the first 800 boosting steps, after which it is allowed to search for interactions (given that other hyperparameters that control interactions also allow this). The motivation for fitting main effects first may be 1) to get a cleaner looking model that puts more emphasis on main effects and 2) to speed up the algorithm since looking for interactions is computationally more demanding.

## Method: fit(X:npt.ArrayLike, y:npt.ArrayLike, sample_weight:npt.ArrayLike = np.empty(0), X_names:List[str]=[], validation_set_indexes:List[int]=[], prioritized_predictors_indexes:List[int]=[], monotonic_constraints:List[int]=[], group:npt.ArrayLike = np.empty(0), interaction_constraints:List[List[int]]=[], other_data: npt.ArrayLike = np.empty([0, 0]))

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

#### group
A numpy vector of integers that is used when ***loss_function*** is "group_mse". For example, ***group*** may represent year (could be useful in a time series model).

#### interaction_constraints
An optional list containing lists of integers. Specifies interaction constraints on model terms. For example, interaction_constraints = [[0,1], [1,2,3]] means that 1) the first and second predictors may interact with each other, and that 2) the second, third and fourth predictors may interact with each other. There are no interaction constraints on predictors not mentioned in interaction_constraints.

#### other_data
An optional numpy matrix with other data. This is used in custom loss, negative gradient and validation error functions.


## Method: predict(X:npt.ArrayLike, cap_predictions_to_minmax_in_training:bool=True)

***Returns a numpy vector containing predictions of the data in X. Requires that the model has been fitted with the fit method.***

### Parameters

#### X
A numpy matrix with predictor values.

#### cap_predictions_to_minmax_in_training
If ***True*** then predictions are capped so that they are not less than the minimum and not greater than the maximum prediction or response in the training dataset. This is recommended especially if ***max_interaction_level*** is high. However, if you need the model to extrapolate then set this parameter to ***False***.


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


## Method: get_optimal_m()

***Returns the number of boosting steps in the model (the value that minimized validation error).***


## Method: get_validation_tuning_metric()

***Returns the validation_tuning_metric used.*** 

## Method: get_validation_indexes()

***Returns a list of integers containing the indexes of the training data observations used for validation and not training.***