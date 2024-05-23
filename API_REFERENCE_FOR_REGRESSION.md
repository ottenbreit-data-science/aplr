# APLRRegressor

## class aplr.APLRRegressor(m:int=3000, v:float=0.1, random_state:int=0, loss_function:str="mse", link_function:str="identity", n_jobs:int=0, cv_folds:int=5, bins:int=300, max_interaction_level:int=1, max_interactions:int=100000, min_observations_in_split:int=20, ineligible_boosting_steps_added:int=10, max_eligible_terms:int=5, verbosity:int=0, dispersion_parameter:float=1.5, validation_tuning_metric:str="default", quantile:float=0.5, calculate_custom_validation_error_function:Optional[Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], float]]=None, calculate_custom_loss_function:Optional[Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], float]]=None, calculate_custom_negative_gradient_function:Optional[Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]]=None, calculate_custom_transform_linear_predictor_to_predictions_function:Optional[Callable[[npt.ArrayLike], npt.ArrayLike]]=None, calculate_custom_differentiate_predictions_wrt_linear_predictor_function:Optional[Callable[[npt.ArrayLike], npt.ArrayLike]]=None, boosting_steps_before_interactions_are_allowed: int = 0, monotonic_constraints_ignore_interactions: bool = False, group_mse_by_prediction_bins: int = 10, group_mse_cycle_min_obs_in_bin: int = 30, early_stopping_rounds: int = 500, num_first_steps_with_linear_effects_only: int = 0,         penalty_for_non_linearity: float = 0.0, penalty_for_interactions: float = 0.0, max_terms: int = 0)

### Constructor parameters

#### m (default = 3000)
The maximum number of boosting steps. If validation error does not flatten out at the end of the ***m***th boosting step, then try increasing it (or alternatively increase the learning rate).

#### v (default = 0.1)
The learning rate. Must be greater than zero and not more than one. The higher the faster the algorithm learns and the lower ***m*** is required. However, empirical evidence suggests that ***v <= 0.1*** gives better results. If the algorithm learns too fast (requires few boosting steps to converge) then try lowering the learning rate. Computational costs can be reduced by increasing the learning rate while simultaneously decreasing ***m***, potentially at the expense of predictiveness.

#### random_state (default = 0)
Used to randomly split training observations into cv_folds if ***cv_observations*** is not specified when fitting.

#### loss_function (default = "mse")
Determines the loss function used. Allowed values are "mse", "binomial", "poisson", "gamma", "tweedie", "group_mse", "group_mse_cycle","mae", "quantile", "negative_binomial", "cauchy", "weibull" and "custom_function". This is used together with ***link_function***. When ***loss_function*** is "group_mse" then the "group" argument in the ***fit*** method must be provided. In the latter case APLR will try to minimize group MSE when training the model. When using "group_mse_cycle", ***group_mse_cycle_min_obs_in_bin*** controls the minimum amount of observations in each group. For a description of "group_mse_cycle" see ***group_mse_cycle_min_obs_in_bin***. The ***loss_function*** "quantile" is used together with the ***quantile*** constructor parameter. When ***loss_function*** is "custom_function" then the constructor parameters ***calculate_custom_loss_function*** and ***calculate_custom_negative_gradient_function***, both described below, must be provided.

#### link_function (default = "identity")
Determines how the linear predictor is transformed to predictions. Allowed values are "identity", "logit", "log" and "custom_function". For an ordinary regression model use ***loss_function*** "mse" and ***link_function*** "identity". For logistic regression use ***loss_function*** "binomial" and ***link_function*** "logit". For a multiplicative model use the "log" ***link_function***. The "log" ***link_function*** often works best with a "poisson", "gamma", "tweedie", "negative_binomial" or "weibull" ***loss_function***, depending on the data. The ***loss_function*** "poisson", "gamma", "tweedie", "negative_binomial" or "weibull" should only be used with the "log" ***link_function***. Inappropriate combinations of ***loss_function*** and ***link_function*** may result in a warning message when fitting the model and/or a poor model fit. Please note that values other than "identity" typically require a significantly higher ***m*** (or ***v***) in order to converge. When ***link_function*** is "custom_function" then the constructor parameters ***calculate_custom_transform_linear_predictor_to_predictions_function*** and ***calculate_custom_differentiate_predictions_wrt_linear_predictor_function***, both described below, must be provided.

#### n_jobs (default = 0)
Multi-threading parameter. If ***0*** then uses all available cores for multi-threading. Any other positive integer specifies the number of cores to use (***1*** means single-threading).

#### cv_folds (default = 5)
The number of randomly split folds to use in cross validation. The number of boosting steps is automatically tuned to minimize cross validation error.

#### bins (default = 300)
Specifies the maximum number of bins to discretize the data into when searching for the best split. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs. Must be greater than 1.

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

#### dispersion_parameter (default = 1.5)
Specifies the variance power when ***loss_function*** is "tweedie". Specifies a dispersion parameter when ***loss_function*** is "negative_binomial", "cauchy" or "weibull". 

#### validation_tuning_metric (default = "default")
Specifies which metric to use for validating the model and tuning ***m***. Available options are "default" (using the same methodology as when calculating the training error), "mse", "mae", "negative_gini", "group_mse", "group_mse_by_prediction" and "custom_function". The default is often a choice that fits well with respect to the ***loss_function*** chosen. However, if you want to use ***loss_function*** or ***dispersion_parameter*** as tuning parameters then the default is not suitable. "group_mse" requires that the "group" argument in the ***fit*** method is provided. "group_mse_by_prediction" groups predictions by up to ***group_mse_by_prediction_bins*** groups and calculates groupwise mse. For "custom_function" see ***calculate_custom_validation_error_function*** below.

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

#### boosting_steps_before_interactions_are_allowed (default = 0)
Specifies how many boosting steps to wait before searching for interactions. If for example 800, then the algorithm will be forced to only fit main effects in the first 800 boosting steps, after which it is allowed to search for interactions (given that other hyperparameters that control interactions also allow this). The motivation for fitting main effects first may be 1) to get a cleaner looking model that puts more emphasis on main effects and 2) to speed up the algorithm since looking for interactions is computationally more demanding.

#### monotonic_constraints_ignore_interactions (default = False)
See ***monotonic_constraints*** in the ***fit*** method.

#### group_mse_by_prediction_bins (default = 10)
Specifies how many groups to bin predictions by when ***validation_tuning_metric*** is "group_mse_by_prediction" (or when ***loss_function*** is "group_mse_cycle" and ***validation_tuning_metric*** is "default").

#### group_mse_cycle_min_obs_in_bin (default = 30)
When ***loss_function*** equals ***group_mse_cycle*** then ***group_mse_cycle_min_obs_in_bin*** specifies the minimum amount of observations in each group. The loss function ***group_mse_cycle*** groups by the first predictor in ***X*** in the first boosting step, then by the second predictor in ***X*** in the second boosting step, etc. So in each boosting step the predictor to group by is changed. If ***validation_tuning_metric*** is "default" then "group_mse_by_prediction" will be used as ***validation_tuning_metric***.

#### early_stopping_rounds (default = 500)
If validation loss does not improve during the last ***early_stopping_rounds*** boosting steps then boosting is aborted. The point with this constructor parameter is to speed up the training and make it easier to select a high ***m***.

#### num_first_steps_with_linear_effects_only (default = 0)
Specifies the number of initial boosting steps that are reserved only for linear effects. 0 means that non-linear effects are allowed from the first boosting step. Reasons for setting this parameter to a higher value than 0 could be to 1) build a more interpretable model with more emphasis on linear effects or 2) build a linear only model by setting ***num_first_steps_with_linear_effects_only*** to no less than ***m***. 

#### penalty_for_non_linearity (default = 0.0)
Specifies a penalty in the range [0.0, 1.0] on terms that are not linear effects. A higher value increases model interpretability but can hurt predictiveness. Values outside of the [0.0, 1.0] range are rounded to the nearest boundary within the range.

#### penalty_for_interactions (default = 0.0)
Specifies a penalty in the range [0.0, 1.0] on interaction terms. A higher value increases model interpretability but can hurt predictiveness. Values outside of the [0.0, 1.0] range are rounded to the nearest boundary within the range.

#### max_terms (default = 0)
Restricts the maximum number of terms in any of the underlying models trained to ***max_terms***. The default value of 0 means no limit. After the limit is reached, the remaining boosting steps are used to further update the coefficients of already included terms. An optional tuning objective could be to find the lowest positive value of ***max_terms*** that does not increase the prediction error significantly. Low positive values can speed up the training process significantly. Setting a limit with ***max_terms*** may require a higher learning rate for best results.


## Method: fit(X:npt.ArrayLike, y:npt.ArrayLike, sample_weight:npt.ArrayLike = np.empty(0), X_names:List[str]=[], cv_observations: npt.ArrayLike = np.empty([0, 0]), prioritized_predictors_indexes:List[int]=[], monotonic_constraints:List[int]=[], group:npt.ArrayLike = np.empty(0), interaction_constraints:List[List[int]]=[], other_data: npt.ArrayLike = np.empty([0, 0]), predictor_learning_rates: List[float] = [], predictor_penalties_for_non_linearity: List[float] = [], predictor_penalties_for_interactions: List[float] = [])

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

#### cv_observations
An optional list of integers specifying how each training observation is used in cross validation. If this is specified then ***cv_folds*** is not used. Specifying ***cv_observations*** may be useful for example when modelling time series data (you can place more recent observations in the holdout folds). ***cv_observations*** must contain a column for each desired fold combination. For a given column, row values equalling 1 specify that these rows will be used for training, while row values equalling -1 specify that these rows will be used for validation. Row values equalling 0 will not be used.

#### prioritized_predictors_indexes
An optional list of integers specifying the indexes of predictors (columns) in ***X*** that should be prioritized. Terms of the prioritized predictors will enter the model as long as they reduce the training error and do not contain too few effective observations. They will also be updated more often.

#### monotonic_constraints
An optional list of integers specifying monotonic constraints on model terms. For example, if there are three predictors in ***X***, then monotonic_constraints = [1,0,-1] means that 1) all terms using the first predictor in ***X*** as a main effect must have positive regression coefficients, 2) there are no monotonic constraints on terms using the second predictor in ***X***, and 3) all terms using the third predictor in ***X*** as a main effect must have negative regression coefficients. In the above example, if ***monotonic_constraints_ignore_interactions*** is ***False*** (default) then the first and the third predictors in ***X*** cannot be used in interaction terms as secondary effects. The latter guarantees monotonicity but can degrade predictiveness especially if a large proportion of predictors have monotonic constraints (in this case significantly fewer interaction terms can be formed).

#### group
A numpy vector of integers that is used when ***loss_function*** is "group_mse". For example, ***group*** may represent year (could be useful in a time series model).

#### interaction_constraints
An optional list containing lists of integers. Specifies interaction constraints on model terms. For example, interaction_constraints = [[0,1], [1,2,3]] means that 1) the first and second predictors may interact with each other, and that 2) the second, third and fourth predictors may interact with each other. There are no interaction constraints on predictors not mentioned in interaction_constraints.

#### other_data
An optional numpy matrix with other data. This is used in custom loss, negative gradient and validation error functions.

#### predictor_learning_rates
An optional list of floats specifying learning rates for each predictor. If provided then this supercedes ***v***. For example, if there are two predictors in ***X***, then predictor_learning_rates = [0.1,0.2] means that all terms using the first predictor in ***X*** as a main effect will have a learning rate of 0.1 and that all terms using the second predictor in ***X*** as a main effect will have a learning rate of 0.2.

#### predictor_penalties_for_non_linearity
An optional list of floats specifying penalties for non-linearity for each predictor. If provided then this supercedes ***penalty_for_non_linearity***. For example, if there are two predictors in ***X***, then predictor_penalties_for_non_linearity = [0.1,0.2] means that all terms using the first predictor in ***X*** as a main effect will have a penalty for non-linearity of 0.1 and that all terms using the second predictor in ***X*** as a main effect will have a penalty for non-linearity of 0.2.

#### predictor_penalties_for_interactions
An optional list of floats specifying interaction penalties for each predictor. If provided then this supercedes ***penalty_for_interactions***. For example, if there are two predictors in ***X***, then predictor_penalties_for_interactions = [0.1,0.2] means that all terms using the first predictor in ***X*** as a main effect will have an interaction penalty of 0.1 and that all terms using the second predictor in ***X*** as a main effect will have an interaction penalty of 0.2.


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


## Method: calculate_feature_importance(X:npt.ArrayLike, sample_weight:npt.ArrayLike = np.empty(0))

***Returns a numpy matrix containing estimated feature importance in X for each predictor.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: calculate_term_importance(X:npt.ArrayLike, sample_weight:npt.ArrayLike = np.empty(0))

***Returns a numpy matrix containing estimated term importance in X for each term in the model.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: calculate_local_feature_contribution(X:npt.ArrayLike)

***Returns a numpy matrix containing estimated feature contribution to the linear predictor in X for each predictor.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: calculate_local_term_contribution(X:npt.ArrayLike)

***Returns a numpy matrix containing term contribution to the linear predictor in X for each term in the model.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: calculate_local_contribution_from_selected_terms(X:npt.ArrayLike, predictor_indexes:List[int])

***Returns a numpy vector containing the contribution to the linear predictor from an user specified combination of interacting predictors for each observation in X. This makes it easier to interpret interactions (or main effects if just one predictor is specified), for example by plotting predictor values against the term contribution.***

### Parameters

#### X
A numpy matrix with predictor values.

#### predictor_indexes
A list of integers specifying the indexes of predictors in X to use. For example, [1, 3] means the second and fourth predictors in X.


## Method: calculate_terms(X:npt.ArrayLike)

***Returns a numpy matrix containing values of model terms calculated on X.***

### Parameters

#### X
A numpy matrix with predictor values.


## Method: get_term_names()

***Returns a list of strings containing term names.***


## Method: get_term_coefficients()

***Returns a numpy vector containing term regression coefficients.***


## Method: get_validation_error_steps()

***Returns a numpy matrix containing the validation error by boosting step for each cv fold. Use this to determine if the maximum number of boosting steps (m) or learning rate (v) should be changed.***


## Method: get_feature_importance()

***Returns a numpy vector containing the estimated feature importance in the training data for each predictor.***


## Method: get_term_importance()

***Returns a numpy vector containing the estimated term importance in the training data for each term.***


## Method: get_term_main_predictor_indexes()

***Returns a numpy vector containing the main predictor index for each term.***


## Method: get_term_interaction_levels()

***Returns a numpy vector containing the interaction level for each term.***


## Method: get_intercept()

***Returns the regression coefficient of the intercept term.***


## Method: get_optimal_m()

***Returns the number of boosting steps in the model (the value that minimized validation error).***


## Method: get_validation_tuning_metric()

***Returns the validation_tuning_metric used.*** 


## Method: get_main_effect_shape(predictor_index:int)

***For the predictor in X specified by predictor_index, get_main_effect_shape returns a dictionary with keys equal to predictor values and values equal to the corresponding contribution to the linear predictor (interactions with other predictors are ignored). This method makes it easier to interpret main effects.***

### Parameters

#### predictor_index
The index of the predictor. So if ***predictor_index*** is ***1*** then the second predictor in ***X*** is used.


## Method: get_cv_error()

***Returns the cv error for the model.***