# The recommended way to interpret an APLR model

## Main effects
Use the ***get_main_effect_shape*** method to interpret main effects as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_regression.py). For each main effect, plot the output in a scatter plot.

## Interactions
For best interpretability of interactions, do not use a higher ***max_interaction_level*** than 1. Use the ***calculate_local_contribution_from_selected_terms*** method to interpret interactions as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_regression.py). For each interaction of interest you can plot the output in a 3D scatter plot.

## Interpretation of model terms and their regression coefficients
For an example on how to interpret the terms in an APLR model, please see ***Section 5.1.3*** in the published article about APLR. You can find this article on [https://link.springer.com/article/10.1007/s00180-024-01475-4](https://link.springer.com/article/10.1007/s00180-024-01475-4) and [https://rdcu.be/dz7bF](https://rdcu.be/dz7bF).