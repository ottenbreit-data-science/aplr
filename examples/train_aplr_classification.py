import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from aplr import APLRClassifier


# Settings
random_state = 0

# Loading data
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = pd.Series(iris.target).astype("str")

# Please note that APLRClassifier requires that all predictor columns in the data have numerical values,
# This means that if you have missing values in the data then you need to either drop rows with missing data or impute them.
# This also means that if you have a categorical text variable then you need to convert it to for example dummy variables for each category.

# However, APLRClassifier requires that the response variable is a list of strings.

# Please also note that APLR may be vulnerable to outliers in predictor values. If you experience this problem then please consider winsorising
# the predictors (or similar methods) before passing them to APLR.

# Randomly splitting data into training and test sets
data_train, data_test = train_test_split(data, test_size=0.3, random_state=random_state)
del data

# Predictors and response
predictors = iris.feature_names
response = "target"
predicted = "predicted"

# Training model
cv_results = pd.DataFrame()
best_validation_result = np.inf
param_grid = ParameterGrid(
    {
        "max_interaction_level": [0, 1],
        "min_observations_in_split": [1, 20, 40],
    }
)
best_model: APLRClassifier = None
for params in param_grid:
    model = APLRClassifier(
        random_state=random_state,
        verbosity=2,
        m=1000,
        v=0.5,
        # max_terms=5,  # Optionally tune this to find a trade-off between interpretability and predictiveness. May require a higher learning rate for best results.
        **params
    )
    model.fit(
        data_train[predictors].values, data_train[response].values, X_names=predictors
    )
    cv_error_for_this_model = model.get_cv_error()  # Based on log loss.
    cv_results_for_this_model = pd.DataFrame(model.get_params(), index=[0])
    cv_results_for_this_model["cv_error"] = cv_error_for_this_model
    cv_results = pd.concat([cv_results, cv_results_for_this_model])
    if cv_error_for_this_model < best_validation_result:
        best_validation_result = cv_error_for_this_model
        best_model = model
print("Done training")

# Saving model
joblib.dump(best_model, "best_model.gz", compress=9)

# Validation results when doing grid search
cv_results = cv_results.sort_values(by="cv_error")

# Validation errors per boosting step for each holdout fold
validation_error_per_boosting_step = best_model.get_validation_error_steps()

# Get a list of the categories in the model
categories = (
    best_model.get_categories()
)  # In this example the categories are "0", "1" and "2".

# Accessing the logit model that predicts whether an observation belongs to class "1" or not. The logit model is an APLRRegressor and can be used
# for example for interpretation purposes.
logit_model_for_class_1 = best_model.get_logit_model(category="1")

# Estimated feature importance in the training data, average of the underlying logit models.
estimated_feature_importance = pd.DataFrame(
    {
        "predictor": best_model.get_unique_term_affiliations(),
        "importance": best_model.get_feature_importance(),
    }
)
estimated_feature_importance = estimated_feature_importance.sort_values(
    by="importance", ascending=False
)

# Local feature contribution for each prediction. For each prediction, uses calculate_local_feature_contribution() in the logit APLRRegressor model
# for the category that corresponds to the prediction. Example in this data: If a prediction is "2" then using calculate_local_feature_contribution()
# in the logit model that predicts whether an observation belongs to class "2" or not. This can be used to interpret the model, for example
# by creating 3D surface plots against predictor values to interpret two-way interactions. This method can also be used on new data.
local_feature_contribution = pd.DataFrame(
    best_model.calculate_local_feature_contribution(data_train[predictors]),
    columns=best_model.get_unique_term_affiliations(),
)


# PREDICTING AND TESTING ON THE TEST SET
data_test[predicted] = best_model.predict(data_test[predictors].values)
predicted_class_probabilities = pd.DataFrame(
    data=best_model.predict_class_probabilities(data_test[predictors].values),
    columns=categories,
)
data_test = pd.concat(
    [data_test.reset_index(drop=True), predicted_class_probabilities], axis=1
)

# Goodness of fit
balanced_accuracy = balanced_accuracy_score(
    y_true=data_test[response], y_pred=data_test[predicted]
)
