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
    size_t reserved_terms_times_num_x;
    std::map<std::string, VectorXd> response_values; // Key is category and value is response vector

    void initialize();
    void find_categories(const std::vector<std::string> &y);
    void create_response_for_each_category(const std::vector<std::string> &y);
    void define_validation_indexes(const std::vector<std::string> &y, const std::vector<size_t> &validation_set_indexes);
    void invert_second_model_in_two_class_case(APLRRegressor &second_model);
    void calculate_validation_metrics();
    void cleanup_after_fit();

public:
    size_t m;
    double v;
    double validation_ratio;
    size_t n_jobs;
    uint_fast32_t random_state;
    size_t bins;
    size_t verbosity;
    size_t max_interaction_level;
    size_t max_interactions;
    size_t min_observations_in_split;
    size_t ineligible_boosting_steps_added;
    size_t max_eligible_terms;
    std::vector<size_t> validation_indexes;
    VectorXd validation_error_steps;
    double validation_error;
    VectorXd feature_importance;
    std::vector<std::string> categories;
    std::map<std::string, APLRRegressor> logit_models; // Key is category and value is logit model
    size_t boosting_steps_before_pruning_is_done;
    size_t boosting_steps_before_interactions_are_allowed;

    APLRClassifier(size_t m = 9000, double v = 0.1, uint_fast32_t random_state = std::numeric_limits<uint_fast32_t>::lowest(), size_t n_jobs = 0,
                   double validation_ratio = 0.2, size_t reserved_terms_times_num_x = 100, size_t bins = 300, size_t verbosity = 0, size_t max_interaction_level = 1,
                   size_t max_interactions = 100000, size_t min_observations_in_split = 20, size_t ineligible_boosting_steps_added = 10, size_t max_eligible_terms = 5,
                   size_t boosting_steps_before_pruning_is_done = 0, size_t boosting_steps_before_interactions_are_allowed = 0);
    APLRClassifier(const APLRClassifier &other);
    ~APLRClassifier();
    void fit(const MatrixXd &X, const std::vector<std::string> &y, const VectorXd &sample_weight = VectorXd(0),
             const std::vector<std::string> &X_names = {}, const std::vector<size_t> &validation_set_indexes = {},
             const std::vector<size_t> &prioritized_predictors_indexes = {}, const std::vector<int> &monotonic_constraints = {},
             const std::vector<std::vector<size_t>> &interaction_constraints = {});
    MatrixXd predict_class_probabilities(const MatrixXd &X, bool cap_predictions_to_minmax_in_training = false);
    std::vector<std::string> predict(const MatrixXd &X, bool cap_predictions_to_minmax_in_training = false);
    MatrixXd calculate_local_feature_importance(const MatrixXd &X);
    std::vector<std::string> get_categories();
    APLRRegressor get_logit_model(const std::string &category);
    std::vector<size_t> get_validation_indexes();
    VectorXd get_validation_error_steps();
    double get_validation_error();
    VectorXd get_feature_importance();
};

APLRClassifier::APLRClassifier(size_t m, double v, uint_fast32_t random_state, size_t n_jobs, double validation_ratio,
                               size_t reserved_terms_times_num_x, size_t bins, size_t verbosity, size_t max_interaction_level, size_t max_interactions,
                               size_t min_observations_in_split, size_t ineligible_boosting_steps_added, size_t max_eligible_terms,
                               size_t boosting_steps_before_pruning_is_done, size_t boosting_steps_before_interactions_are_allowed)
    : m{m}, v{v}, random_state{random_state}, n_jobs{n_jobs}, validation_ratio{validation_ratio},
      reserved_terms_times_num_x{reserved_terms_times_num_x}, bins{bins}, verbosity{verbosity}, max_interaction_level{max_interaction_level},
      max_interactions{max_interactions}, min_observations_in_split{min_observations_in_split},
      ineligible_boosting_steps_added{ineligible_boosting_steps_added}, max_eligible_terms{max_eligible_terms},
      boosting_steps_before_pruning_is_done{boosting_steps_before_pruning_is_done},
      boosting_steps_before_interactions_are_allowed{boosting_steps_before_interactions_are_allowed}
{
}

APLRClassifier::APLRClassifier(const APLRClassifier &other)
    : m{other.m}, v{other.v}, random_state{other.random_state}, n_jobs{other.n_jobs}, validation_ratio{other.validation_ratio},
      reserved_terms_times_num_x{other.reserved_terms_times_num_x}, bins{other.bins}, verbosity{other.verbosity},
      max_interaction_level{other.max_interaction_level}, max_interactions{other.max_interactions},
      min_observations_in_split{other.min_observations_in_split}, ineligible_boosting_steps_added{other.ineligible_boosting_steps_added},
      max_eligible_terms{other.max_eligible_terms}, logit_models{other.logit_models}, categories{other.categories},
      validation_indexes{other.validation_indexes}, validation_error_steps{other.validation_error_steps}, validation_error{other.validation_error},
      feature_importance{other.feature_importance}, boosting_steps_before_pruning_is_done{other.boosting_steps_before_pruning_is_done},
      boosting_steps_before_interactions_are_allowed{other.boosting_steps_before_interactions_are_allowed}
{
}

APLRClassifier::~APLRClassifier()
{
}

void APLRClassifier::fit(const MatrixXd &X, const std::vector<std::string> &y, const VectorXd &sample_weight, const std::vector<std::string> &X_names,
                         const std::vector<size_t> &validation_set_indexes, const std::vector<size_t> &prioritized_predictors_indexes,
                         const std::vector<int> &monotonic_constraints, const std::vector<std::vector<size_t>> &interaction_constraints)
{
    initialize();
    find_categories(y);
    create_response_for_each_category(y);
    define_validation_indexes(y, validation_set_indexes);

    bool two_class_case{categories.size() == 2};
    if (two_class_case)
    {
        logit_models[categories[0]] = APLRRegressor(m, v, random_state, "binomial", "logit", n_jobs, validation_ratio, reserved_terms_times_num_x,
                                                    bins, verbosity, max_interaction_level, max_interactions, min_observations_in_split, ineligible_boosting_steps_added,
                                                    max_eligible_terms, 1.5, "default", 0.5);
        logit_models[categories[0]].boosting_steps_before_pruning_is_done = boosting_steps_before_pruning_is_done;
        logit_models[categories[0]].boosting_steps_before_interactions_are_allowed = boosting_steps_before_interactions_are_allowed;
        logit_models[categories[0]].fit(X, response_values[categories[0]], sample_weight, X_names, validation_indexes, prioritized_predictors_indexes,
                                        monotonic_constraints, VectorXi(0), interaction_constraints);

        logit_models[categories[1]] = logit_models[categories[0]];
        invert_second_model_in_two_class_case(logit_models[categories[1]]);
    }
    else
    {
        for (auto &category : categories)
        {
            logit_models[category] = APLRRegressor(m, v, random_state, "binomial", "logit", n_jobs, validation_ratio, reserved_terms_times_num_x,
                                                   bins, verbosity, max_interaction_level, max_interactions, min_observations_in_split, ineligible_boosting_steps_added,
                                                   max_eligible_terms, 1.5, "default", 0.5);
            logit_models[category].boosting_steps_before_pruning_is_done = boosting_steps_before_pruning_is_done;
            logit_models[category].boosting_steps_before_interactions_are_allowed = boosting_steps_before_interactions_are_allowed;
            logit_models[category].fit(X, response_values[category], sample_weight, X_names, validation_indexes, prioritized_predictors_indexes,
                                       monotonic_constraints, VectorXi(0), interaction_constraints);
        }
    }

    calculate_validation_metrics();
    cleanup_after_fit();
}

void APLRClassifier::initialize()
{
    logit_models.clear();
    categories.clear();
    validation_indexes.clear();
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

void APLRClassifier::define_validation_indexes(const std::vector<std::string> &y, const std::vector<size_t> &validation_set_indexes)
{
    bool validation_set_indexes_is_not_provided{validation_set_indexes.size() == 0};
    if (validation_set_indexes_is_not_provided)
    {
        validation_indexes = std::vector<size_t>(0);
        validation_indexes.reserve(y.size());
        std::mt19937 mersenne{random_state};
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double roll;
        for (size_t i = 0; i < y.size(); ++i)
        {
            roll = distribution(mersenne);
            bool place_in_validation_set{std::isless(roll, validation_ratio)};
            if (place_in_validation_set)
            {
                validation_indexes.push_back(i);
            }
        }
        validation_indexes.shrink_to_fit();
    }
    else
        validation_indexes = validation_set_indexes;
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

void APLRClassifier::calculate_validation_metrics()
{
    double category_weight{1.0 / static_cast<double>(categories.size())};
    validation_error_steps = VectorXd::Constant(m, 0.0);
    validation_error = 0;
    feature_importance = VectorXd::Constant(logit_models[categories[0]].get_feature_importance().rows(), 0.0);
    for (std::string &category : categories)
    {
        validation_error += logit_models[category].get_validation_error_steps().minCoeff() * category_weight;
        for (size_t row = 0; row < m; ++row)
        {
            validation_error_steps[row] += logit_models[category].get_validation_error_steps()[row] * category_weight;
        }
        feature_importance += logit_models[category].get_feature_importance() * category_weight;
    }
}

void APLRClassifier::cleanup_after_fit()
{
    response_values.clear();
}

MatrixXd APLRClassifier::predict_class_probabilities(const MatrixXd &X, bool cap_predictions_to_minmax_in_training)
{
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

MatrixXd APLRClassifier::calculate_local_feature_importance(const MatrixXd &X)
{
    MatrixXd output{MatrixXd::Constant(X.rows(), feature_importance.rows(), 0)};
    std::vector<std::string> predictions{predict(X, false)};
    for (size_t row = 0; row < predictions.size(); ++row)
    {
        output.row(row) = logit_models[predictions[row]].calculate_local_feature_importance(X.row(row));
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

std::vector<size_t> APLRClassifier::get_validation_indexes()
{
    return validation_indexes;
}

VectorXd APLRClassifier::get_validation_error_steps()
{
    return validation_error_steps;
}

double APLRClassifier::get_validation_error()
{
    return validation_error;
}

VectorXd APLRClassifier::get_feature_importance()
{
    return feature_importance;
}