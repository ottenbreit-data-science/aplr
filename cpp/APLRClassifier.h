#pragma once
#include <string>
#include <vector>
#include <map>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "APLRRegressor.h"
#include "functions.h"
#include "constants.h"

using namespace Eigen;

class APLRClassifier
{
private:
    std::map<std::string, VectorXd> response_values; // Key is category and value is response vector

    void initialize();
    void find_categories(const std::vector<std::string> &y);
    void create_response_for_each_category(const std::vector<std::string> &y);
    void define_cv_observations(const std::vector<std::string> &y, const MatrixXi &cv_observations_);
    void invert_second_model_in_two_class_case(APLRRegressor &second_model);
    void calculate_validation_metrics();
    void calculate_unique_term_affiliations();
    void throw_error_if_not_fitted();
    void cleanup_after_fit();

public:
    size_t m;
    double v;
    size_t cv_folds;
    size_t n_jobs;
    uint_fast32_t random_state;
    size_t bins;
    size_t verbosity;
    size_t max_interaction_level;
    size_t max_interactions;
    size_t min_observations_in_split;
    size_t ineligible_boosting_steps_added;
    size_t max_eligible_terms;
    MatrixXi cv_observations;
    MatrixXd validation_error_steps;
    double cv_error;
    VectorXd feature_importance;
    std::vector<std::string> categories;
    std::map<std::string, APLRRegressor> logit_models; // Key is category and value is logit model
    size_t boosting_steps_before_interactions_are_allowed;
    bool monotonic_constraints_ignore_interactions;
    size_t early_stopping_rounds;
    size_t num_first_steps_with_linear_effects_only;
    double penalty_for_non_linearity;
    double penalty_for_interactions;
    size_t max_terms;
    std::vector<std::string> unique_term_affiliations;
    std::map<std::string, size_t> unique_term_affiliation_map;
    std::vector<std::vector<size_t>> base_predictors_in_each_unique_term_affiliation;
    double ridge_penalty;

    APLRClassifier(size_t m = 3000, double v = 0.5, uint_fast32_t random_state = std::numeric_limits<uint_fast32_t>::lowest(), size_t n_jobs = 0,
                   size_t cv_folds = 5, size_t bins = 300, size_t verbosity = 0, size_t max_interaction_level = 1,
                   size_t max_interactions = 100000, size_t min_observations_in_split = 4, size_t ineligible_boosting_steps_added = 15, size_t max_eligible_terms = 7,
                   size_t boosting_steps_before_interactions_are_allowed = 0, bool monotonic_constraints_ignore_interactions = false,
                   size_t early_stopping_rounds = 200, size_t num_first_steps_with_linear_effects_only = 0,
                   double penalty_for_non_linearity = 0.0, double penalty_for_interactions = 0.0, size_t max_terms = 0, double ridge_penalty = 0.0001);
    APLRClassifier(const APLRClassifier &other);
    ~APLRClassifier();
    void fit(const MatrixXd &X, const std::vector<std::string> &y, const VectorXd &sample_weight = VectorXd(0),
             const std::vector<std::string> &X_names = {}, const MatrixXi &cv_observations = MatrixXi(0, 0),
             const std::vector<size_t> &prioritized_predictors_indexes = {}, const std::vector<int> &monotonic_constraints = {},
             const std::vector<std::vector<size_t>> &interaction_constraints = {}, const std::vector<double> &predictor_learning_rates = {},
             const std::vector<double> &predictor_penalties_for_non_linearity = {},
             const std::vector<double> &predictor_penalties_for_interactions = {},
             const std::vector<size_t> &predictor_min_observations_in_split = {});
    MatrixXd predict_class_probabilities(const MatrixXd &X, bool cap_predictions_to_minmax_in_training = false);
    std::vector<std::string> predict(const MatrixXd &X, bool cap_predictions_to_minmax_in_training = false);
    MatrixXd calculate_local_feature_contribution(const MatrixXd &X);
    std::vector<std::string> get_categories();
    APLRRegressor get_logit_model(const std::string &category);
    MatrixXd get_validation_error_steps();
    double get_cv_error();
    VectorXd get_feature_importance();
    std::vector<std::string> get_unique_term_affiliations();
    std::vector<std::vector<size_t>> get_base_predictors_in_each_unique_term_affiliation();
    void clear_cv_results();
};

APLRClassifier::APLRClassifier(size_t m, double v, uint_fast32_t random_state, size_t n_jobs, size_t cv_folds,
                               size_t bins, size_t verbosity, size_t max_interaction_level, size_t max_interactions,
                               size_t min_observations_in_split, size_t ineligible_boosting_steps_added, size_t max_eligible_terms,
                               size_t boosting_steps_before_interactions_are_allowed, bool monotonic_constraints_ignore_interactions,
                               size_t early_stopping_rounds, size_t num_first_steps_with_linear_effects_only,
                               double penalty_for_non_linearity, double penalty_for_interactions, size_t max_terms, double ridge_penalty)
    : m{m}, v{v}, random_state{random_state}, n_jobs{n_jobs}, cv_folds{cv_folds},
      bins{bins}, verbosity{verbosity}, max_interaction_level{max_interaction_level},
      max_interactions{max_interactions}, min_observations_in_split{min_observations_in_split},
      ineligible_boosting_steps_added{ineligible_boosting_steps_added}, max_eligible_terms{max_eligible_terms},
      boosting_steps_before_interactions_are_allowed{boosting_steps_before_interactions_are_allowed},
      monotonic_constraints_ignore_interactions{monotonic_constraints_ignore_interactions}, early_stopping_rounds{early_stopping_rounds},
      num_first_steps_with_linear_effects_only{num_first_steps_with_linear_effects_only}, penalty_for_non_linearity{penalty_for_non_linearity},
      penalty_for_interactions{penalty_for_interactions}, max_terms{max_terms}, ridge_penalty{ridge_penalty}
{
}

APLRClassifier::APLRClassifier(const APLRClassifier &other)
    : m{other.m}, v{other.v}, random_state{other.random_state}, n_jobs{other.n_jobs}, cv_folds{other.cv_folds},
      bins{other.bins}, verbosity{other.verbosity},
      max_interaction_level{other.max_interaction_level}, max_interactions{other.max_interactions},
      min_observations_in_split{other.min_observations_in_split}, ineligible_boosting_steps_added{other.ineligible_boosting_steps_added},
      max_eligible_terms{other.max_eligible_terms}, logit_models{other.logit_models}, categories{other.categories},
      cv_observations{other.cv_observations}, validation_error_steps{other.validation_error_steps}, cv_error{other.cv_error},
      feature_importance{other.feature_importance},
      boosting_steps_before_interactions_are_allowed{other.boosting_steps_before_interactions_are_allowed},
      monotonic_constraints_ignore_interactions{other.monotonic_constraints_ignore_interactions},
      early_stopping_rounds{other.early_stopping_rounds},
      num_first_steps_with_linear_effects_only{other.num_first_steps_with_linear_effects_only},
      penalty_for_non_linearity{other.penalty_for_non_linearity}, penalty_for_interactions{other.penalty_for_interactions},
      max_terms{other.max_terms}, unique_term_affiliations{other.unique_term_affiliations},
      unique_term_affiliation_map{other.unique_term_affiliation_map},
      base_predictors_in_each_unique_term_affiliation{other.base_predictors_in_each_unique_term_affiliation},
      ridge_penalty{other.ridge_penalty}
{
}

APLRClassifier::~APLRClassifier()
{
}

void APLRClassifier::fit(const MatrixXd &X, const std::vector<std::string> &y, const VectorXd &sample_weight, const std::vector<std::string> &X_names,
                         const MatrixXi &cv_observations, const std::vector<size_t> &prioritized_predictors_indexes,
                         const std::vector<int> &monotonic_constraints, const std::vector<std::vector<size_t>> &interaction_constraints,
                         const std::vector<double> &predictor_learning_rates, const std::vector<double> &predictor_penalties_for_non_linearity,
                         const std::vector<double> &predictor_penalties_for_interactions,
                         const std::vector<size_t> &predictor_min_observations_in_split)
{
    initialize();
    find_categories(y);
    create_response_for_each_category(y);
    define_cv_observations(y, cv_observations);

    bool two_class_case{categories.size() == 2};
    if (two_class_case)
    {
        logit_models[categories[0]] = APLRRegressor(m, v, random_state, "binomial", "logit", n_jobs, cv_folds,
                                                    bins, verbosity, max_interaction_level, max_interactions, min_observations_in_split, ineligible_boosting_steps_added,
                                                    max_eligible_terms, 1.5, "default", 0.5);
        logit_models[categories[0]].boosting_steps_before_interactions_are_allowed = boosting_steps_before_interactions_are_allowed;
        logit_models[categories[0]].monotonic_constraints_ignore_interactions = monotonic_constraints_ignore_interactions;
        logit_models[categories[0]].early_stopping_rounds = early_stopping_rounds;
        logit_models[categories[0]].num_first_steps_with_linear_effects_only = num_first_steps_with_linear_effects_only;
        logit_models[categories[0]].penalty_for_non_linearity = penalty_for_non_linearity;
        logit_models[categories[0]].penalty_for_interactions = penalty_for_interactions;
        logit_models[categories[0]].max_terms = max_terms;
        logit_models[categories[0]].ridge_penalty = ridge_penalty;
        logit_models[categories[0]].fit(X, response_values[categories[0]], sample_weight, X_names, cv_observations, prioritized_predictors_indexes,
                                        monotonic_constraints, VectorXi(0), interaction_constraints, MatrixXd(0, 0), predictor_learning_rates,
                                        predictor_penalties_for_non_linearity, predictor_penalties_for_interactions,
                                        predictor_min_observations_in_split);

        logit_models[categories[1]] = logit_models[categories[0]];
        invert_second_model_in_two_class_case(logit_models[categories[1]]);
    }
    else
    {
        for (auto &category : categories)
        {
            logit_models[category] = APLRRegressor(m, v, random_state, "binomial", "logit", n_jobs, cv_folds,
                                                   bins, verbosity, max_interaction_level, max_interactions, min_observations_in_split, ineligible_boosting_steps_added,
                                                   max_eligible_terms, 1.5, "default", 0.5);
            logit_models[category].boosting_steps_before_interactions_are_allowed = boosting_steps_before_interactions_are_allowed;
            logit_models[category].monotonic_constraints_ignore_interactions = monotonic_constraints_ignore_interactions;
            logit_models[category].early_stopping_rounds = early_stopping_rounds;
            logit_models[category].num_first_steps_with_linear_effects_only = num_first_steps_with_linear_effects_only;
            logit_models[category].penalty_for_non_linearity = penalty_for_non_linearity;
            logit_models[category].penalty_for_interactions = penalty_for_interactions;
            logit_models[category].max_terms = max_terms;
            logit_models[category].ridge_penalty = ridge_penalty;
            logit_models[category].fit(X, response_values[category], sample_weight, X_names, cv_observations, prioritized_predictors_indexes,
                                       monotonic_constraints, VectorXi(0), interaction_constraints, MatrixXd(0, 0), predictor_learning_rates,
                                       predictor_penalties_for_non_linearity, predictor_penalties_for_interactions,
                                       predictor_min_observations_in_split);
        }
    }

    calculate_unique_term_affiliations();
    calculate_validation_metrics();
    cleanup_after_fit();
}

void APLRClassifier::initialize()
{
    logit_models.clear();
    categories.clear();
    cv_observations.resize(0, 0);
}

void APLRClassifier::find_categories(const std::vector<std::string> &y)
{
    std::set<std::string> set_of_categories{get_unique_strings(y)};
    bool too_few_categories{set_of_categories.size() < MIN_CATEGORIES_IN_CLASSIFIER};
    if (too_few_categories)
        throw std::runtime_error("The number of categories must be at least " + std::to_string(MIN_CATEGORIES_IN_CLASSIFIER) + ".");

    categories.reserve(set_of_categories.size());
    for (auto &category : set_of_categories)
    {
        categories.push_back(category);
    }
}

void APLRClassifier::create_response_for_each_category(const std::vector<std::string> &y)
{
    for (auto &category : categories)
    {
        response_values[category] = VectorXd::Constant(y.size(), 0.0);
        for (size_t i = 0; i < y.size(); ++i)
        {
            if (y[i] == category)
                response_values[category][i] = 1.0;
        }
    }
}

void APLRClassifier::define_cv_observations(const std::vector<std::string> &y, const MatrixXi &cv_observations_)
{
    APLRRegressor aplr_regressor{APLRRegressor(m, v, random_state, "binomial", "logit", n_jobs, cv_folds,
                                               bins, verbosity, max_interaction_level, max_interactions, min_observations_in_split, ineligible_boosting_steps_added,
                                               max_eligible_terms, 1.5, "default", 0.5)};
    VectorXd y_dummy_vector{VectorXd(y.size())};
    cv_observations = aplr_regressor.preprocess_cv_observations(cv_observations_, y_dummy_vector);
}

void APLRClassifier::invert_second_model_in_two_class_case(APLRRegressor &second_model)
{
    second_model.intercept = -second_model.intercept;
    for (Term &term : second_model.terms)
    {
        term.coefficient = -term.coefficient;
        for (double &coefficient_steps : term.coefficient_steps)
        {
            coefficient_steps = -coefficient_steps;
        }
    }
    for (double &coefficient : second_model.term_coefficients)
    {
        coefficient = -coefficient;
    }
}

void APLRClassifier::calculate_unique_term_affiliations()
{
    size_t number_of_term_affiliations{0};
    for (std::string &category : categories)
    {
        number_of_term_affiliations += logit_models[category].number_of_unique_term_affiliations;
    }
    std::vector<std::string> term_affiliations;
    term_affiliations.reserve(number_of_term_affiliations);
    size_t counter{0};
    for (std::string &category : categories)
    {
        for (auto &affiliation : logit_models[category].unique_term_affiliations)
        {
            term_affiliations.push_back(affiliation);
            ++counter;
        }
    }
    unique_term_affiliations = get_unique_strings_as_vector(term_affiliations);
    for (size_t i = 0; i < unique_term_affiliations.size(); ++i)
    {
        unique_term_affiliation_map[unique_term_affiliations[i]] = i;
    }
    base_predictors_in_each_unique_term_affiliation.resize(unique_term_affiliation_map.size());
    std::vector<std::set<size_t>> base_predictors_in_each_unique_term_affiliation_set(unique_term_affiliation_map.size());
    for (std::string &category : categories)
    {
        for (auto &term : logit_models[category].terms)
        {
            std::vector<size_t> unique_base_terms_for_this_term{term.get_unique_base_terms_used_in_this_term()};
            base_predictors_in_each_unique_term_affiliation_set[unique_term_affiliation_map[term.predictor_affiliation]].insert(unique_base_terms_for_this_term.begin(), unique_base_terms_for_this_term.end());
        }
    }
    for (size_t i = 0; i < base_predictors_in_each_unique_term_affiliation_set.size(); ++i)
    {
        base_predictors_in_each_unique_term_affiliation[i] = std::vector<size_t>(base_predictors_in_each_unique_term_affiliation_set[i].begin(), base_predictors_in_each_unique_term_affiliation_set[i].end());
    }
}

void APLRClassifier::calculate_validation_metrics()
{
    double category_weight{1.0 / static_cast<double>(categories.size())};
    validation_error_steps = MatrixXd::Constant(m, cv_observations.cols(), 0.0);
    cv_error = 0.0;
    feature_importance = VectorXd::Constant(unique_term_affiliations.size(), 0.0);
    for (std::string &category : categories)
    {
        cv_error += logit_models[category].get_cv_error() * category_weight;
        validation_error_steps += logit_models[category].get_validation_error_steps() * category_weight;
        for (auto &affiliation : logit_models[category].unique_term_affiliations)
        {
            size_t feature_number_in_classifier{unique_term_affiliation_map[affiliation]};
            size_t feature_number_in_logit_model{logit_models[category].unique_term_affiliation_map[affiliation]};
            feature_importance[feature_number_in_classifier] += logit_models[category].get_feature_importance()[feature_number_in_logit_model] * category_weight;
        }
    }
}

void APLRClassifier::cleanup_after_fit()
{
    response_values.clear();
}

void APLRClassifier::throw_error_if_not_fitted()
{
    if (categories.empty())
    {
        throw std::runtime_error("This APLRClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.");
    }
}

MatrixXd APLRClassifier::predict_class_probabilities(const MatrixXd &X, bool cap_predictions_to_minmax_in_training)
{
    throw_error_if_not_fitted();

    MatrixXd predictions{MatrixXd::Constant(X.rows(), categories.size(), 0.0)};
    for (size_t i = 0; i < categories.size(); ++i)
    {
        predictions.col(i) = logit_models[categories[i]].predict(X, cap_predictions_to_minmax_in_training);
    }

    for (size_t row = 0; row < predictions.rows(); ++row)
    {
        double rowsum{predictions.row(row).sum()};
        for (size_t col = 0; col < predictions.cols(); ++col)
        {
            predictions.row(row)[col] /= rowsum;
        }
    }

    return predictions;
}

std::vector<std::string> APLRClassifier::predict(const MatrixXd &X, bool cap_predictions_to_minmax_in_training)
{
    throw_error_if_not_fitted();
    std::vector<std::string> predictions(X.rows());
    MatrixXd predicted_class_probabilities{predict_class_probabilities(X, cap_predictions_to_minmax_in_training)};
    for (size_t row = 0; row < predicted_class_probabilities.rows(); ++row)
    {
        size_t best_category_index;
        predicted_class_probabilities.row(row).maxCoeff(&best_category_index);
        predictions[row] = categories[best_category_index];
    }

    return predictions;
}

MatrixXd APLRClassifier::calculate_local_feature_contribution(const MatrixXd &X)
{
    throw_error_if_not_fitted();
    MatrixXd output{MatrixXd::Constant(X.rows(), unique_term_affiliations.size(), 0)};
    std::vector<std::string> predictions{predict(X, false)};
    for (size_t row = 0; row < predictions.size(); ++row)
    {
        VectorXd local_feature_contribution_from_logit_model{logit_models[predictions[row]].calculate_local_feature_contribution(X.row(row)).row(0)};
        for (auto &affiliation : logit_models[predictions[row]].unique_term_affiliations)
        {
            size_t feature_number_in_classifier{unique_term_affiliation_map[affiliation]};
            size_t feature_number_in_logit_model{logit_models[predictions[row]].unique_term_affiliation_map[affiliation]};
            output.col(feature_number_in_classifier)[row] = local_feature_contribution_from_logit_model[feature_number_in_logit_model];
        }
    }

    return output;
}

std::vector<std::string> APLRClassifier::get_categories()
{
    return categories;
}

APLRRegressor APLRClassifier::get_logit_model(const std::string &category)
{
    bool category_does_not_exist{true};
    for (auto &available_category : categories)
    {
        if (category == available_category)
        {
            category_does_not_exist = false;
            break;
        }
    }
    if (category_does_not_exist)
        throw std::runtime_error("Invalid category provided.");

    return logit_models[category];
}

MatrixXd APLRClassifier::get_validation_error_steps()
{
    return validation_error_steps;
}

double APLRClassifier::get_cv_error()
{
    return cv_error;
}

VectorXd APLRClassifier::get_feature_importance()
{
    return feature_importance;
}

std::vector<std::string> APLRClassifier::get_unique_term_affiliations()
{
    return unique_term_affiliations;
}

std::vector<std::vector<size_t>> APLRClassifier::get_base_predictors_in_each_unique_term_affiliation()
{
    return base_predictors_in_each_unique_term_affiliation;
}

void APLRClassifier::clear_cv_results()
{
    throw_error_if_not_fitted();
    for (auto &pair : logit_models)
    {
        pair.second.clear_cv_results();
    }
}