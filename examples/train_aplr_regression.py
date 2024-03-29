import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.datasets import load_diabetes
from aplr import APLRRegressor


# Settings
random_state = 0

# Loading data
diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data["target"] = pd.Series(diabetes.target)

# Please note that APLR requires that all columns in the data have numerical values.
# This means that if you have missing values in the data then you need to either drop rows with missing data or impute them.
# This also means that if you have a categorical text variable then you need to convert it to for example dummy variables for each category.

# Please also note that APLR may be vulnerable to outliers in predictor values. If you experience this problem then please consider winsorising
# the predictors (or similar methods) before passing them to APLR.

# Randomly splitting data into training and test sets
data_train, data_test = train_test_split(data, test_size=0.3, random_state=random_state)
del data

# Predictors and response
predictors = diabetes.feature_names
response = "target"
predicted = "predicted"

# Training model
cv_results = pd.DataFrame()
best_validation_result = np.inf
param_grid = ParameterGrid(
    {
        "max_interaction_level": [0, 1, 2, 3, 100],
        "min_observations_in_split": [1, 20, 50, 100, 200],
    }
)
best_model = None
loss_function = "mse"  # other available loss functions are binomial, poisson, gamma, tweedie, group_mse, mae, quantile, negative_binomial, cauchy, weibull and custom_function.
link_function = (
    "identity"  # other available link functions are logit, log and custom_function.
)
for params in param_grid:
    model = APLRRegressor(
        random_state=random_state,
        verbosity=2,
        m=3000,
        v=0.1,
        loss_function=loss_function,
        link_function=link_function,
        **params
    )
    model.fit(
        data_train[predictors].values, data_train[response].values, X_names=predictors
    )
    cv_error_for_this_model = model.get_cv_error()
    cv_results_for_this_model = pd.DataFrame(model.get_params(), index=[0])
    cv_results_for_this_model["cv_error"] = cv_error_for_this_model
    cv_results = pd.concat([cv_results, cv_results_for_this_model])
    if cv_error_for_this_model < best_validation_result:
        best_validation_result = cv_error_for_this_model
        best_model = model
print("Done training")

# Saving model
joblib.dump(best_model, "best_model.gz")

# Validation results when doing grid search
cv_results = cv_results.sort_values(by="cv_error")

# Validation errors that occurred during training of the best model for each holdout fold.
# APLR used the boosting steps that gave the lowest validation errors for each fold.
validation_error_per_boosting_step = best_model.get_validation_error_steps()

# Terms in the best model
terms = pd.DataFrame(
    {
        "term": best_model.get_term_names(),
        "coefficient": best_model.get_term_coefficients(),
    }
)

# Estimated feature importance in the training data
estimated_feature_importance = pd.DataFrame(
    {"predictor": predictors, "importance": best_model.get_feature_importance()}
)
estimated_feature_importance = estimated_feature_importance.sort_values(
    by="importance", ascending=False
)

# Estimated term importance in the training data
term_names_excluding_intercept = best_model.get_term_names()[1:]
estimated_term_importance = pd.DataFrame(
    {
        "term": term_names_excluding_intercept,
        "importance": best_model.get_term_importance(),
    }
)

# Coefficient shape for the third predictor. Will be empty if the third predictor is not used as a main effect in the model
coefficient_shape = best_model.get_coefficient_shape_function(predictor_index=2)
coefficient_shape = pd.DataFrame(
    {
        "predictor_value": coefficient_shape.keys(),
        "coefficient": coefficient_shape.values(),
    }
)


# PREDICTING AND TESTING ON THE TEST SET
data_test[predicted] = best_model.predict(data_test[predictors].values)

# Goodness of fit
correlation = pd.DataFrame(
    {"response": data_test[response], "prediction": data_test[predicted]}
).corr()
mse = ((data_test[response] - data_test[predicted]) ** 2).mean()
mae = (data_test[response] - data_test[predicted]).abs().mean()
goodness_of_fit = pd.DataFrame(
    {"mse": [mse], "mae": [mae], "correlation": [correlation["prediction"][0]]}
)
goodness_of_fit["r_squared"] = goodness_of_fit["correlation"] ** 2

# Estimated feature importance in the test set
estimated_feature_importance_in_test_set = pd.DataFrame(
    {
        "predictor": predictors,
        "importance": best_model.calculate_feature_importance(data_test[predictors]),
    }
)
estimated_feature_importance_in_test_set = (
    estimated_feature_importance_in_test_set.sort_values(
        by="importance", ascending=False
    )
)

# Estimated term importance in the test set
estimated_term_importance_in_test_set = pd.DataFrame(
    {
        "term": term_names_excluding_intercept,
        "importance": best_model.calculate_term_importance(data_test[predictors]),
    }
)
estimated_term_importance_in_test_set = (
    estimated_term_importance_in_test_set.sort_values(by="importance", ascending=False)
)


# Local contribution for each prediction in the test set
estimated_local_feature_contribution = pd.DataFrame(
    best_model.calculate_local_feature_contribution(data_test[predictors]),
    columns=predictors,
)
local_term_contribution = pd.DataFrame(
    best_model.calculate_local_term_contribution(data_test[predictors]),
    columns=term_names_excluding_intercept,
)


# Calculate terms on test data
calculated_terms = pd.DataFrame(
    best_model.calculate_terms(data_test[predictors]),
    columns=term_names_excluding_intercept,
)
