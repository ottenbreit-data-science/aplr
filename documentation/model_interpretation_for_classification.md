# The recommended way to interpret an APLR model for classification

## Feature importance
Use the ***get_feature_importance*** method as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_classification.py).

## Main effects and interactions
For best interpretability of interactions, do not use a higher ***max_interaction_level*** than 1. Use the ***calculate_local_feature_contribution*** method to interpret main effects and interactions as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_classification.py). You may also use the ***get_logit_model*** method to access the underlying APLR regression models as shown in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_classification.py). You can interpret these models in the same way as described in this [example](https://github.com/ottenbreit-data-science/aplr/blob/main/examples/train_aplr_regression.py).