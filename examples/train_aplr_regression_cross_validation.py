import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_diabetes
from aplr import APLRRegressor


# Settings
random_state = 0

# Loading data
diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data["target"] = pd.Series(diabetes.target)

# Please note that the approach described in train_aplr_regression_validation.py is usually significantly faster.

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
param_grid = {
    "max_interaction_level": [0, 1, 2, 3, 100],
    "min_observations_in_split": [1, 20, 50, 100, 200],
}
loss_function = "mse"  # other available families are binomial, poisson, gamma, tweedie, group_mse, mae, quantile, negative_binomial, cauchy and weibull.
link_function = "identity"  # other available link functions are logit and log.
grid_search_cv = GridSearchCV(
    APLRRegressor(
        random_state=random_state,
        verbosity=1,
        m=1000,
        v=0.01,
        loss_function=loss_function,
        link_function=link_function,
    ),
    param_grid,
    cv=5,
    n_jobs=4,
    scoring="neg_mean_squared_error",
)
grid_search_cv.fit(data_train[predictors].values, data_train[response].values)
best_model: APLRRegressor = grid_search_cv.best_estimator_
best_model.set_term_names(X_names=predictors)
print("Done training")

# Saving model
joblib.dump(best_model, "best_model.gz")

# Cross validation results when doing grid search
cv_results = pd.DataFrame(grid_search_cv.cv_results_).sort_values(by="rank_test_score")

# Validation errors that occurred during training of the best model. APLR used the boosting step that gave the lowest validation error
validation_error_per_boosting_step = best_model.get_validation_error_steps()

# Terms in the best model
terms = pd.DataFrame(
    {
        "term": best_model.get_term_names(),
        "coefficient": best_model.get_term_coefficients(),
    }
)

# Coefficients for intercept and the first term per boosting step
intercept_coefficient_per_boosting_step = best_model.get_intercept_steps()
first_term_coefficient_per_boosting_step = best_model.get_term_coefficient_steps(
    term_index=0
)

# Estimated feature importance was estimated on the validation set when the best model was trained
estimated_feature_importance = pd.DataFrame(
    {"predictor": predictors, "importance": best_model.get_feature_importance()}
)
estimated_feature_importance = estimated_feature_importance.sort_values(
    by="importance", ascending=False
)

# Coefficient shape for the first predictor. Will be empty if the first predictor is not used as a main effect in the model
coefficient_shape = best_model.get_coefficient_shape_function(predictor_index=0)
coefficient_shape = pd.DataFrame(
    {"predictor_value": coefficient_shape.keys(), "coefficient": coefficient_shape.values()}
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

# Local feature importance for each prediction
term_names_excluding_intercept = best_model.get_term_names()[1:]
local_feature_importance_of_each_term = pd.DataFrame(
    best_model.calculate_local_feature_importance_for_terms(data_test[predictors]),
    columns=term_names_excluding_intercept,
)
estimated_local_feature_importance_of_each_original_predictor = pd.DataFrame(
    best_model.calculate_local_feature_importance(data_test[predictors]),
    columns=predictors,
)

# Calculate terms on test data
calculated_terms = pd.DataFrame(
    best_model.calculate_terms(data_test[predictors]),
    columns=term_names_excluding_intercept,
)
