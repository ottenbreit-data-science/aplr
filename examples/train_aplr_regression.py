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
        "max_interaction_level": [0, 1],
        "min_observations_in_split": [1, 20, 50, 100, 200],
    }
)
best_model: APLRRegressor = None
loss_function = "mse"  # Other available loss functions are binomial, poisson, gamma, tweedie, group_mse, mae, quantile, negative_binomial, cauchy, weibull and custom_function.
link_function = (
    "identity"  # Other available link functions are logit, log and custom_function.
)
for params in param_grid:
    model = APLRRegressor(
        random_state=random_state,
        verbosity=2,
        m=3000,
        v=0.1,
        loss_function=loss_function,
        link_function=link_function,
        # max_terms=10,  # Optionally tune this to find a trade-off between interpretability and predictiveness. May require a higher learning rate for best results.
        **params,
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
joblib.dump(best_model, "best_model.gz", compress=9)

# Validation results when doing grid search
cv_results = cv_results.sort_values(by="cv_error")

# Validation errors that occurred during training of the best model for each holdout fold.
# APLR used the boosting steps that gave the lowest validation errors for each fold.
validation_error_per_boosting_step = best_model.get_validation_error_steps()

# Terms in the best model
terms = pd.DataFrame(
    {
        "predictor_affiliation": ["Intercept"] + best_model.get_term_affiliations(),
        "term": best_model.get_term_names(),
        "coefficient": best_model.get_term_coefficients(),
        "estimated_term_importance": np.concatenate(
            (np.zeros(1), best_model.get_term_importance())
        ),
    }
)

# Estimated feature importance in the training data
estimated_feature_importance = pd.DataFrame(
    {
        "predictor": best_model.get_unique_term_affiliations(),
        "importance": best_model.get_feature_importance(),
    }
)
estimated_feature_importance = estimated_feature_importance.sort_values(
    by="importance", ascending=False
)

# Main effect shape for the third predictor. This can be visualized in a scatter plot.
# Will be empty if the third predictor is not used as a main effect in the model.
main_effect_shape = best_model.get_main_effect_shape(predictor_index=2)
main_effect_shape = pd.DataFrame(
    {
        "predictor_value": main_effect_shape.keys(),
        "contribution_to_linear_predictor": main_effect_shape.values(),
    }
)

# Local contribution to the linear predictor for each prediction in the training data. This can be used to interpret the model,
# for example by visualizing two-way interactions versus predictor values in a 3D scatter plot. This method can also be used on new data.
local_feature_contribution = pd.DataFrame(
    best_model.calculate_local_feature_contribution(data_train[predictors]),
    columns=best_model.get_unique_term_affiliations(),
)
# Combining predictor values with local feature contribution for the second feature in best_model.get_unique_term_affiliations().
# This can be visualized if it is a main effect or a two-way interaction.
unique_term_affiliation_index = 1
predictors_in_the_second_feature = [
    predictors[predictor_index]
    for predictor_index in best_model.get_base_predictors_in_each_unique_term_affiliation()[
        unique_term_affiliation_index
    ]
]
data_to_visualize = pd.DataFrame(
    np.concatenate(
        (
            data_train[predictors_in_the_second_feature].values,
            local_feature_contribution[
                [
                    best_model.get_unique_term_affiliations()[
                        unique_term_affiliation_index
                    ]
                ]
            ],
        ),
        axis=1,
    ),
    columns=predictors_in_the_second_feature
    + [
        f"contribution from {best_model.get_unique_term_affiliations()[unique_term_affiliation_index]}"
    ],
)

# Local (observation specific) contribution to the linear predictor from selected interacting predictors.
# In this example this concerns two-way interaction terms in the model where the fourth and the seventh predictors in X interact.
# The local contribution will be zero for all observations if there are no such terms in the model.
# The local contribution can help interpreting interactions (or main effects if only one predictor index is specified).
# For two-way interactions the local contribution can be plotted against the predictor values in a 3D scatter plot.
contribution_from_selected_terms = (
    best_model.calculate_local_contribution_from_selected_terms(
        X=data_train[predictors], predictor_indexes=[3, 6]
    )
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
        "predictor": best_model.get_unique_term_affiliations(),
        "importance": best_model.calculate_feature_importance(data_test[predictors]),
    }
)
estimated_feature_importance_in_test_set = (
    estimated_feature_importance_in_test_set.sort_values(
        by="importance", ascending=False
    )
)

# Estimated term importance in the test set
term_names_excluding_intercept = best_model.get_term_names()[1:]
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
local_term_contribution = pd.DataFrame(
    best_model.calculate_local_term_contribution(data_test[predictors]),
    columns=term_names_excluding_intercept,
)

# Calculate terms on test data
calculated_terms = pd.DataFrame(
    best_model.calculate_terms(data_test[predictors]),
    columns=term_names_excluding_intercept,
)
