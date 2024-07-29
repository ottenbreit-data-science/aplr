# APLRClassifier

## class aplr.APLRClassifier(m:int = 20000, v:float = 0.5, random_state:int = 0, n_jobs:int = 0, cv_folds:int = 5, bins:int = 300, verbosity:int = 0, max_interaction_level:int = 1, max_interactions:int = 100000, min_observations_in_split:int = 4, ineligible_boosting_steps_added:int = 15, max_eligible_terms:int = 5, boosting_steps_before_interactions_are_allowed: int = 0, monotonic_constraints_ignore_interactions: bool = False, early_stopping_rounds: int = 500, num_first_steps_with_linear_effects_only: int = 0, penalty_for_non_linearity: float = 0.0, penalty_for_interactions: float = 0.0, max_terms: int = 0)

### Constructor parameters

#### m (default = 20000)
The maximum number of boosting steps. If validation error does not flatten out at the end of the ***m***th boosting step, then try increasing it (or alternatively increase the learning rate).

#### v (default = 0.5)
The learning rate. Must be greater than zero and not more than one. The higher the faster the algorithm learns and the lower ***m*** is required, reducing computational costs potentially at the expense of predictiveness. Empirical evidence suggests that ***v <= 0.5*** gives good results for APLR.

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
Specifies the maximum allowed depth of interaction terms. ***0*** means that interactions are not allowed. This hyperparameter should be tuned by for example doing a grid search for best predictiveness. For best interpretability use 0 (or 1 if interactions are needed).

#### max_interactions (default = 100000)
The maximum number of interactions allowed in each underlying model. A lower value may be used to reduce computational time or to increase interpretability.

#### min_observations_in_split (default = 4)
The minimum effective number of observations that a term in the model must rely on. This hyperparameter should be tuned. Larger values are more appropriate for larger datasets. Larger values result in more robust models (lower variance), potentially at the expense of increased bias.

#### ineligible_boosting_steps_added (default = 15)
Controls how many boosting steps a term that becomes ineligible has to remain ineligible. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs.

#### max_eligible_terms (default = 5)
Limits 1) the number of terms already in the model that can be considered as interaction partners in a boosting step and 2) how many terms remain eligible in the next boosting step. The default value works well according to empirical results. This hyperparameter is intended for reducing computational costs.

#### boosting_steps_before_interactions_are_allowed (default = 0)
Specifies how many boosting steps to wait before searching for interactions. If for example 800, then the algorithm will be forced to only fit main effects in the first 800 boosting steps, after which it is allowed to search for interactions (given that other hyperparameters that control interactions also allow this). The motivation for fitting main effects first may be 1) to get a cleaner looking model that puts more emphasis on main effects and 2) to speed up the algorithm since looking for interactions is computationally more demanding.

#### monotonic_constraints_ignore_interactions (default = False)
See ***monotonic_constraints*** in the ***fit*** method.

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


## Method: fit(X:FloatMatrix, y:List[str], sample_weight:FloatVector = np.empty(0), X_names:List[str] = [], cv_observations:IntMatrix = np.empty([0, 0]), prioritized_predictors_indexes:List[int] = [], monotonic_constraints:List[int] = [], interaction_constraints:List[List[int]] = [], predictor_learning_rates:List[float] = [], predictor_penalties_for_non_linearity:List[float] = [], predictor_penalties_for_interactions:List[float] = [])

***This method fits the model to data.***

### Parameters

#### X
A numpy matrix with predictor values.

#### y
A list of strings with response values (class names).

#### sample_weight
An optional numpy vector with sample weights. If not specified then the observations are weighted equally.

#### X_names
An optional list of strings containing names for each predictor in ***X***. Naming predictors may increase model readability because model terms get names based on ***X_names***.

#### cv_observations
An optional integer matrix specifying how each training observation is used in cross validation. If this is specified then ***cv_folds*** is not used. Specifying ***cv_observations*** may be useful for example when modelling time series data (you can place more recent observations in the holdout folds). ***cv_observations*** must contain a column for each desired fold combination. For a given column, row values equalling 1 specify that these rows will be used for training, while row values equalling -1 specify that these rows will be used for validation. Row values equalling 0 will not be used.

#### prioritized_predictors_indexes
An optional list of integers specifying the indexes of predictors (columns) in ***X*** that should be prioritized. Terms of the prioritized predictors will enter the model as long as they reduce the training error and do not contain too few effective observations. They will also be updated more often.

#### monotonic_constraints
An optional list of integers specifying monotonic constraints on model terms. For example, if there are three predictors in ***X***, then monotonic_constraints = [1,0,-1] means that 1) all terms using the first predictor in ***X*** as a main effect must have positive regression coefficients, 2) there are no monotonic constraints on terms using the second predictor in ***X***, and 3) all terms using the third predictor in ***X*** as a main effect must have negative regression coefficients. In the above example, if ***monotonic_constraints_ignore_interactions*** is ***False*** (default) then the first and the third predictors in ***X*** cannot be used in interaction terms as secondary effects. The latter guarantees monotonicity but can degrade predictiveness especially if a large proportion of predictors have monotonic constraints (in this case significantly fewer interaction terms can be formed).

#### interaction_constraints
An optional list containing lists of integers. Specifies interaction constraints on model terms. For example, interaction_constraints = [[0,1], [1,2,3]] means that 1) the first and second predictors may interact with each other, and that 2) the second, third and fourth predictors may interact with each other. There are no interaction constraints on predictors not mentioned in interaction_constraints.

#### predictor_learning_rates
An optional list of floats specifying learning rates for each predictor. If provided then this supercedes ***v***. For example, if there are two predictors in ***X***, then predictor_learning_rates = [0.1, 0.2] means that all terms using the first predictor in ***X*** as a main effect will have a learning rate of 0.1 and that all terms using the second predictor in ***X*** as a main effect will have a learning rate of 0.2.

#### predictor_penalties_for_non_linearity
An optional list of floats specifying penalties for non-linearity for each predictor. If provided then this supercedes ***penalty_for_non_linearity***. For example, if there are two predictors in ***X***, then predictor_penalties_for_non_linearity = [0.1,0.2] means that all terms using the first predictor in ***X*** as a main effect will have a penalty for non-linearity of 0.1 and that all terms using the second predictor in ***X*** as a main effect will have a penalty for non-linearity of 0.2.

#### predictor_penalties_for_interactions
An optional list of floats specifying interaction penalties for each predictor. If provided then this supercedes ***penalty_for_interactions***. For example, if there are two predictors in ***X***, then predictor_penalties_for_interactions = [0.1,0.2] means that all terms using the first predictor in ***X*** as a main effect will have an interaction penalty of 0.1 and that all terms using the second predictor in ***X*** as a main effect will have an interaction penalty of 0.2.


## Method: predict_class_probabilities(X:FloatMatrix, cap_predictions_to_minmax_in_training:bool = False)

***Returns a numpy matrix containing predictions of the data in X. Requires that the model has been fitted with the fit method.***

### Parameters

#### X
A numpy matrix with predictor values.

#### cap_predictions_to_minmax_in_training
If ***True*** then for each underlying logit model the predictions are capped so that they are not less than the minimum and not greater than the maximum prediction or response in the training dataset.


## Method: predict(X:FloatMatrix, cap_predictions_to_minmax_in_training:bool = False)

***Returns a list of strings containing predictions of the data in X. An observation is classified to the category with the highest predicted class probability. Requires that the model has been fitted with the fit method.***

### Parameters
Parameters are the same as in ***predict_class_probabilities()***.


## Method: calculate_local_feature_contribution(X:FloatMatrix)

***Returns a numpy matrix containing feature contribution to the linear predictor in X for each predictor. For each prediction this method uses calculate_local_feature_contribution() in the logit APLRRegressor model for the category that corresponds to the prediction. Example: If a prediction is "myclass" then the method uses calculate_local_feature_contribution() in the logit model that predicts whether an observation belongs to class "myclass" or not.***

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


## Method: get_unique_term_affiliations()

***Returns a list of strings containing unique predictor affiliations for terms.***


## Method: get_base_predictors_in_each_unique_term_affiliation()

***Returns a list of integer lists. The first list contains indexes for the unique base predictors used in the first unique term affiliation. The second list contains indexes for the unique base predictors used in the second unique term affiliation, and so on.***