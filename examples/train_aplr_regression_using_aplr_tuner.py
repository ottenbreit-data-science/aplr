import pandas as pd
import numpy as np
from typing import Dict
import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from aplr import APLRTuner


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
loss_function = "mse"  # Other available loss functions are binomial, poisson, gamma, tweedie, group_mse, mae, quantile, negative_binomial, cauchy, weibull and custom_function.
link_function = (
    "identity"  # Other available link functions are logit, log and custom_function.
)
parameters = {
    "random_state": [random_state],
    "max_interaction_level": [0, 1],
    "min_observations_in_split": [1, 4, 20, 50],
    "verbosity": [2],
    "m": [3000],
    "v": [0.5],
    "loss_function": [loss_function],
    "link_function": [link_function],
    "ridge_penalty": [0, 0.0001, 0.001],
    "num_first_steps_with_linear_effects_only": [
        0
    ],  # Increasing num_first_steps_with_linear_effects_only will increase interpretabilty but may decrease predictiveness.
    "boosting_steps_before_interactions_are_allowed": [
        0
    ],  # Increasing boosting_steps_before_interactions_are_allowed will increase interpretabilty but may decrease predictiveness.
}
aplr_tuner = APLRTuner(parameters=parameters, is_regressor=True)
aplr_tuner.fit(
    X=data_train[predictors].values, y=data_train[response].values, X_names=predictors
)
best_model = aplr_tuner.get_best_estimator()
cv_results = pd.DataFrame(aplr_tuner.get_cv_results())
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

# Shapes for all term affiliations in the model. For each term affiliation, shape_df contains predictor values and the corresponding
# contributions to the linear predictor. Plots are created for main effects and two-way interactions.
# This is probably the most useful method to use for understanding how the model works.
predictors_in_each_affiliation = (
    best_model.get_base_predictors_in_each_unique_term_affiliation()
)
for affiliation_index, affiliation in enumerate(
    best_model.get_unique_term_affiliations()
):
    shape = best_model.get_unique_term_affiliation_shape(affiliation)
    predictor_indexes_used = predictors_in_each_affiliation[affiliation_index]
    shape_df = pd.DataFrame(
        shape,
        columns=[predictors[i] for i in predictor_indexes_used] + ["contribution"],
    )
    is_main_effect: bool = len(predictor_indexes_used) == 1
    is_two_way_interaction: bool = len(predictor_indexes_used) == 2
    if is_main_effect:
        plt.plot(shape_df.iloc[:, 0], shape_df.iloc[:, 1])
        plt.xlabel(shape_df.columns[0])
        plt.ylabel(shape_df.columns[1])
        plt.title("Contribution to the linear predictor")
        plt.savefig(f"shape of {affiliation}.png")
        plt.close()
    elif is_two_way_interaction:
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_trisurf(
            shape_df.iloc[:, 0],
            shape_df.iloc[:, 1],
            shape_df.iloc[:, 2],
            cmap="Greys",
        )
        ax.set_xlabel(shape_df.columns[0])
        ax.set_ylabel(shape_df.columns[1])
        ax.set_zlabel("contribution")
        plt.title("Contribution to the linear predictor")
        plt.savefig(f"shape of {affiliation}.png")
        plt.close()

# Main effect shape for the third predictor. This can be visualized in a line plot.
# Will be empty if the third predictor is not used as a main effect in the model.
main_effect_shape = best_model.get_main_effect_shape(predictor_index=2)
main_effect_shape = pd.DataFrame(
    {
        "predictor_value": main_effect_shape.keys(),
        "contribution_to_linear_predictor": main_effect_shape.values(),
    }
)

# Local contribution to the linear predictor for each prediction in the training data. This can be used to interpret the model,
# for example by visualizing two-way interactions versus predictor values in a 3D surface plot. This method can also be used on new data.
local_feature_contribution = pd.DataFrame(
    best_model.calculate_local_feature_contribution(data_train[predictors]),
    columns=best_model.get_unique_term_affiliations(),
)

# Local (observation specific) contribution to the linear predictor from selected interacting predictors.
# In this example this concerns two-way interaction terms in the model where the fourth and the seventh predictors in X interact.
# The local contribution will be zero for all observations if there are no such terms in the model.
# The local contribution can help interpreting interactions (or main effects if only one predictor index is specified).
# For two-way interactions the local contribution can be plotted against the predictor values in a 3D surface plot.
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
