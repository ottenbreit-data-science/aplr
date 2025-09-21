# The recommended way to interpret an APLR model for classification

## Feature importance
Use the ***get_feature_importance*** method as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_classification.py).

## Local feature contribution
Use the ***calculate_local_feature_contribution*** method, for example on test data or new data. Usage of this method is demonstrated in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_classification.py).

## Main effects and interactions
For each category, you can interpret the main effects and interactions of its underlying logit model. For best interpretability of interactions, do not use a higher ***max_interaction_level*** than 1.

A convenient way to visualize the model components is to first use the ***get_logit_model*** method to access the underlying `APLRRegressor` model for a specific category. Then, you can use the ***plot_affiliation_shape*** method on that logit model to generate plots for its main effects (line plots) and two-way interactions (heatmaps). This is demonstrated in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_classification.py).