# The recommended way to interpret an APLR model for regression

## Feature importance
Use the ***get_feature_importance*** method as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_regression.py).

## Local feature importance and contribution
Use the ***calculate_feature_importance*** method or the ***calculate_local_feature_contribution*** method, for example on test data or new data. Usage of these methods is demonstrated in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_regression.py).

## Main effects
Use the ***plot_affiliation_shape*** method to easily plot main effects, as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_regression.py). Alternatively, use the ***get_main_effect_shape*** or ***get_unique_term_affiliation_shape*** methods to get the data for a custom plot.

## Interactions
For best interpretability of interactions, do not use a higher ***max_interaction_level*** than 1. Use the ***plot_affiliation_shape*** method to easily plot two-way interactions as a heatmap, as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_regression.py). Alternatively, use the ***get_unique_term_affiliation_shape*** method to get the data for a custom plot, for example a 3D surface plot.

## Interpretation of model terms and their regression coefficients
The above interpretations of main effects and interactions are sufficient to interpret an APLR model. However, it is possible to also inspect the underlying terms for those who wish to do so. For an example on how to interpret the terms in an APLR model, please see ***Section 5.1.3*** in the published article about APLR. You can find this article on [https://link.springer.com/article/10.1007/s00180-024-01475-4](https://link.springer.com/article/10.1007/s00180-024-01475-4) and [https://rdcu.be/dz7bF](https://rdcu.be/dz7bF).