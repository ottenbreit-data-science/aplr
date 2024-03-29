# APLRClassifier

## class aplr.APLRClassifier(m:int=3000, v:float=0.1, random_state:int=0, n_jobs:int=0, cv_folds:int=5, bins:int=300, verbosity:int=0, max_interaction_level:int=1, max_interactions:int=100000, min_observations_in_split:int=20, ineligible_boosting_steps_added:int=10, max_eligible_terms:int=5, boosting_steps_before_interactions_are_allowed: int = 0, monotonic_constraints_ignore_interactions: bool = False, early_stopping_rounds: int = 500)

### Constructor parameters

#### m (default = 3000)
The maximum number of boosting steps. If validation error does not flatten out at the end of the ***m***th boosting step, then try increasing it (or alternatively increase the learning rate).

#### v (default = 0.1)
The learning rate. Must be greater than zero and not more than one. The higher the faster the algorithm learns and the lower ***m*** is required. However, empirical evidence suggests that ***v <= 0.1*** gives better results. If the algorithm learns too fast (requires few boosting steps to converge) then try lowering the learning rate. Computational costs can be reduced by increasing the learning rate while simultaneously decreasing ***m***, potentially at the expense of predictiveness.

#### random_state (default = 0)
Used to randomly split training observations into cv_folds if ***cv_observations*** is not specified when fitting.

#### n_jobs (default = 0)
Multi-threading parameter. If ***0*** then uses all available cores for multi-threading. Any other positive integer specifies the number of cores to use (***1*** means single-threading).

#### cv_folds (default = 5)
The number of randomly split folds to use in cross validation. The number of boosting steps is automatically tuned to minimize cross validation error.

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

#### boosting_steps_before_interactions_are_allowed (default = 0)
Specifies how many boosting steps to wait before searching for interactions. If for example 800, then the algorithm will be forced to only fit main effects in the first 800 boosting steps, after which it is allowed to search for interactions (given that other hyperparameters that control interactions also allow this). The motivation for fitting main effects first may be 1) to get a cleaner looking model that puts more emphasis on main effects and 2) to speed up the algorithm since looking for interactions is computationally more demanding.

#### monotonic_constraints_ignore_interactions (default = False)
See ***monotonic_constraints*** in the ***fit*** method.

#### early_stopping_rounds (default = 500)
If validation loss does not improve during the last ***early_stopping_rounds*** boosting steps then boosting is aborted. The point with this constructor parameter is to speed up the training and make it easier to select a high ***m***.


## Method: fit(X:npt.ArrayLike, y:List[str], sample_weight:npt.ArrayLike = np.empty(0), X_names:List[str]=[], cv_observations: npt.ArrayLike = np.empty([0, 0]), prioritized_predictors_indexes:List[int]=[], monotonic_constraints:List[int]=[], interaction_constraints:List[List[int]]=[])

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

#### interaction_constraints
An optional list containing lists of integers. Specifies interaction constraints on model terms. For example, interaction_constraints = [[0,1], [1,2,3]] means that 1) the first and second predictors may interact with each other, and that 2) the second, third and fourth predictors may interact with each other. There are no interaction constraints on predictors not mentioned in interaction_constraints.


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


## Method: calculate_local_feature_contribution(X:npt.ArrayLike)

***Returns a numpy matrix containing estimated feature contribution to the linear predictor in X for each predictor.***

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


## Method: get_validation_error_steps()

***Returns a numpy matrix containing the validation error by boosting step for each cv fold (average of log loss for each underlying logit model). Use this to determine if the maximum number of boosting steps (m) or learning rate (v) should be changed.***


## Method: get_cv_error()

***Returns the cv error measured by the average of log loss for each underlying logit model.***


## Method: get_feature_importance()

***Returns a numpy vector containing the feature importance of each predictor, estimated as an average of feature importances for the underlying logit models.***