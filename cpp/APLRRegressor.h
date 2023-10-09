#pragma once
#include <string>
#include <limits>
#include <thread>
#include <future>
#include <random>
#include <vector>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "functions.h"
#include "term.h"
#include "constants.h"

using namespace Eigen;

class APLRRegressor
{
private:
    size_t reserved_terms_times_num_x;
    MatrixXd X_train;
    VectorXd y_train;
    VectorXd sample_weight_train;
    MatrixXd X_validation;
    VectorXd y_validation;
    VectorXd sample_weight_validation;
    VectorXd linear_predictor_null_model;
    std::vector<Term> terms_eligible_current;
    VectorXd predictions_current;
    VectorXd predictions_current_validation;
    VectorXd neg_gradient_current;
    double neg_gradient_nullmodel_errors_sum;
    size_t best_term_index;
    VectorXd linear_predictor_update;
    VectorXd linear_predictor_update_validation;
    size_t number_of_eligible_terms;
    std::vector<std::vector<size_t>> distributed_terms;
    std::vector<Term> interactions_to_consider;
    VectorXi sorted_indexes_of_errors_for_interactions_to_consider;
    bool abort_boosting;
    VectorXd linear_predictor_current;
    VectorXd linear_predictor_current_validation;
    double scaling_factor_for_log_link_function;
    std::vector<size_t> predictor_indexes;
    std::vector<size_t> prioritized_predictors_indexes;
    std::vector<int> monotonic_constraints;
    VectorXi group_train;
    VectorXi group_validation;
    std::set<int> unique_groups_train;
    std::set<int> unique_groups_validation;
    std::vector<std::vector<size_t>> interaction_constraints;
    bool pruning_was_done_in_the_current_boosting_step;
    MatrixXd other_data_train;
    MatrixXd other_data_validation;

    void validate_input_to_fit(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight, const std::vector<std::string> &X_names,
                               const std::vector<size_t> &validation_set_indexes, const std::vector<size_t> &prioritized_predictors_indexes,
                               const std::vector<int> &monotonic_constraints, const VectorXi &group, const std::vector<std::vector<size_t>> &interaction_constraints,
                               const MatrixXd &other_data);
    void throw_error_if_validation_set_indexes_has_invalid_indexes(const VectorXd &y, const std::vector<size_t> &validation_set_indexes);
    void throw_error_if_prioritized_predictors_indexes_has_invalid_indexes(const MatrixXd &X, const std::vector<size_t> &prioritized_predictors_indexes);
    void throw_error_if_monotonic_constraints_has_invalid_indexes(const MatrixXd &X, const std::vector<int> &monotonic_constraints);
    void throw_error_if_interaction_constraints_has_invalid_indexes(const MatrixXd &X, const std::vector<std::vector<size_t>> &interaction_constraints);
    void define_training_and_validation_sets(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight,
                                             const std::vector<size_t> &validation_set_indexes, const VectorXi &group, const MatrixXd &other_data);
    void initialize(const std::vector<size_t> &prioritized_predictors_indexes, const std::vector<int> &monotonic_constraints,
                    const std::vector<std::vector<size_t>> &interaction_constraints);
    bool check_if_base_term_has_only_one_unique_value(size_t base_term);
    void add_term_to_terms_eligible_current(Term &term);
    VectorXd calculate_neg_gradient_current(const VectorXd &sample_weight_train);
    void execute_boosting_steps();
    void execute_boosting_step(size_t boosting_step);
    std::vector<size_t> find_terms_eligible_current_indexes_for_a_base_term(size_t base_term);
    void estimate_split_point_for_each_term(std::vector<Term> &terms, std::vector<size_t> &terms_indexes);
    size_t find_best_term_index(std::vector<Term> &terms, std::vector<size_t> &terms_indexes);
    void consider_interactions(const std::vector<size_t> &available_predictor_indexes, size_t boosting_step);
    void determine_interactions_to_consider(const std::vector<size_t> &available_predictor_indexes);
    VectorXi find_indexes_for_terms_to_consider_as_interaction_partners();
    size_t find_out_how_many_terms_to_consider_as_interaction_partners();
    void add_necessary_given_terms_to_interaction(Term &interaction, Term &existing_model_term);
    void find_sorted_indexes_for_errors_for_interactions_to_consider();
    void add_promising_interactions_and_select_the_best_one();
    void update_intercept(size_t boosting_step);
    void select_the_best_term_and_update_errors(size_t boosting_step);
    void remove_ineligibility();
    void update_terms(size_t boosting_step);
    void update_gradient_and_errors();
    void add_new_term(size_t boosting_step);
    void prune_terms(size_t boosting_step);
    void calculate_and_validate_validation_error(size_t boosting_step);
    void calculate_validation_error(size_t boosting_step, const VectorXd &predictions);
    void update_term_eligibility();
    void print_summary_after_boosting_step(size_t boosting_step);
    void update_coefficients_for_all_steps();
    void print_final_summary();
    void find_optimal_m_and_update_model_accordingly();
    void merge_similar_terms();
    void remove_unused_terms();
    void name_terms(const MatrixXd &X, const std::vector<std::string> &X_names);
    void calculate_feature_importance_on_validation_set();
    void find_min_and_max_training_predictions_or_responses();
    void cleanup_after_fit();
    void validate_that_model_can_be_used(const MatrixXd &X);
    void throw_error_if_loss_function_does_not_exist();
    void throw_error_if_link_function_does_not_exist();
    VectorXd calculate_linear_predictor(const MatrixXd &X);
    void update_linear_predictor_and_predictions();
    void throw_error_if_response_contains_invalid_values(const VectorXd &y);
    void throw_error_if_sample_weight_contains_invalid_values(const VectorXd &y, const VectorXd &sample_weight);
    void throw_error_if_response_is_not_between_0_and_1(const VectorXd &y, const std::string &error_message);
    void throw_error_if_vector_contains_negative_values(const VectorXd &y, const std::string &error_message);
    void throw_error_if_response_is_not_greater_than_zero(const VectorXd &y, const std::string &error_message);
    void throw_error_if_dispersion_parameter_is_invalid();
    VectorXd differentiate_predictions_wrt_linear_predictor();
    void scale_training_observations_if_using_log_link_function();
    void revert_scaling_if_using_log_link_function();
    void cap_predictions_to_minmax_in_training(VectorXd &predictions);
    std::string compute_raw_base_term_name(const Term &term, const std::string &X_name);
    void throw_error_if_m_is_invalid();
    bool model_has_not_been_trained();

public:
    double intercept;
    std::vector<Term> terms;
    size_t m;
    size_t m_optimal;
    double v;
    std::string loss_function;
    std::string link_function;
    double validation_ratio;
    size_t n_jobs;
    uint_fast32_t random_state;
    size_t bins;
    size_t verbosity;
    std::vector<std::string> term_names;
    VectorXd term_coefficients;
    size_t max_interaction_level;
    VectorXd intercept_steps;
    size_t max_interactions;
    size_t interactions_eligible;
    VectorXd validation_error_steps;
    size_t min_observations_in_split;
    size_t ineligible_boosting_steps_added;
    size_t max_eligible_terms;
    size_t number_of_base_terms;
    VectorXd feature_importance;
    double dispersion_parameter;
    double min_training_prediction_or_response;
    double max_training_prediction_or_response;
    std::vector<size_t> validation_indexes;
    std::string validation_tuning_metric;
    double quantile;
    std::function<double(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group, const MatrixXd &other_data)> calculate_custom_validation_error_function;
    std::function<double(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group, const MatrixXd &other_data)> calculate_custom_loss_function;
    std::function<VectorXd(const VectorXd &y, const VectorXd &predictions, const VectorXi &group, const MatrixXd &other_data)> calculate_custom_negative_gradient_function;
    std::function<VectorXd(const VectorXd &linear_predictor)> calculate_custom_transform_linear_predictor_to_predictions_function;
    std::function<VectorXd(const VectorXd &linear_predictor)> calculate_custom_differentiate_predictions_wrt_linear_predictor_function;
    size_t boosting_steps_before_pruning_is_done;
    size_t boosting_steps_before_interactions_are_allowed;

    APLRRegressor(size_t m = 1000, double v = 0.1, uint_fast32_t random_state = std::numeric_limits<uint_fast32_t>::lowest(), std::string loss_function = "mse",
                  std::string link_function = "identity", size_t n_jobs = 0, double validation_ratio = 0.2,
                  size_t reserved_terms_times_num_x = 100, size_t bins = 300, size_t verbosity = 0, size_t max_interaction_level = 1, size_t max_interactions = 100000,
                  size_t min_observations_in_split = 20, size_t ineligible_boosting_steps_added = 10, size_t max_eligible_terms = 5, double dispersion_parameter = 1.5,
                  std::string validation_tuning_metric = "default", double quantile = 0.5,
                  const std::function<double(VectorXd, VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_validation_error_function = {},
                  const std::function<double(VectorXd, VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_loss_function = {},
                  const std::function<VectorXd(VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_negative_gradient_function = {},
                  const std::function<VectorXd(VectorXd)> &calculate_custom_transform_linear_predictor_to_predictions_function = {},
                  const std::function<VectorXd(VectorXd)> &calculate_custom_differentiate_predictions_wrt_linear_predictor_function = {},
                  size_t boosting_steps_before_pruning_is_done = 0, size_t boosting_steps_before_interactions_are_allowed = 0);
    APLRRegressor(const APLRRegressor &other);
    ~APLRRegressor();
    void fit(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight = VectorXd(0), const std::vector<std::string> &X_names = {},
             const std::vector<size_t> &validation_set_indexes = {}, const std::vector<size_t> &prioritized_predictors_indexes = {},
             const std::vector<int> &monotonic_constraints = {}, const VectorXi &group = VectorXi(0), const std::vector<std::vector<size_t>> &interaction_constraints = {},
             const MatrixXd &other_data = MatrixXd(0, 0));
    VectorXd predict(const MatrixXd &X, bool cap_predictions_to_minmax_in_training = true);
    void set_term_names(const std::vector<std::string> &X_names);
    MatrixXd calculate_local_feature_importance(const MatrixXd &X);
    MatrixXd calculate_local_feature_importance_for_terms(const MatrixXd &X);
    MatrixXd calculate_terms(const MatrixXd &X);
    std::vector<std::string> get_term_names();
    VectorXd get_term_coefficients();
    VectorXd get_term_coefficient_steps(size_t term_index);
    VectorXd get_validation_error_steps();
    VectorXd get_feature_importance();
    double get_intercept();
    VectorXd get_intercept_steps();
    size_t get_optimal_m();
    std::string get_validation_tuning_metric();
    std::vector<size_t> get_validation_indexes();
};

APLRRegressor::APLRRegressor(size_t m, double v, uint_fast32_t random_state, std::string loss_function, std::string link_function, size_t n_jobs,
                             double validation_ratio, size_t reserved_terms_times_num_x, size_t bins, size_t verbosity, size_t max_interaction_level,
                             size_t max_interactions, size_t min_observations_in_split, size_t ineligible_boosting_steps_added, size_t max_eligible_terms, double dispersion_parameter,
                             std::string validation_tuning_metric, double quantile,
                             const std::function<double(VectorXd, VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_validation_error_function,
                             const std::function<double(VectorXd, VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_loss_function,
                             const std::function<VectorXd(VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_negative_gradient_function,
                             const std::function<VectorXd(VectorXd)> &calculate_custom_transform_linear_predictor_to_predictions_function,
                             const std::function<VectorXd(VectorXd)> &calculate_custom_differentiate_predictions_wrt_linear_predictor_function,
                             size_t boosting_steps_before_pruning_is_done, size_t boosting_steps_before_interactions_are_allowed)
    : reserved_terms_times_num_x{reserved_terms_times_num_x}, intercept{NAN_DOUBLE}, m{m}, v{v},
      loss_function{loss_function}, link_function{link_function}, validation_ratio{validation_ratio}, n_jobs{n_jobs}, random_state{random_state},
      bins{bins}, verbosity{verbosity}, max_interaction_level{max_interaction_level}, intercept_steps{VectorXd(0)},
      max_interactions{max_interactions}, interactions_eligible{0}, validation_error_steps{VectorXd(0)},
      min_observations_in_split{min_observations_in_split}, ineligible_boosting_steps_added{ineligible_boosting_steps_added},
      max_eligible_terms{max_eligible_terms}, number_of_base_terms{0}, dispersion_parameter{dispersion_parameter}, min_training_prediction_or_response{NAN_DOUBLE},
      max_training_prediction_or_response{NAN_DOUBLE}, validation_tuning_metric{validation_tuning_metric},
      validation_indexes{std::vector<size_t>(0)}, quantile{quantile}, calculate_custom_validation_error_function{calculate_custom_validation_error_function},
      calculate_custom_loss_function{calculate_custom_loss_function}, calculate_custom_negative_gradient_function{calculate_custom_negative_gradient_function},
      calculate_custom_transform_linear_predictor_to_predictions_function{calculate_custom_transform_linear_predictor_to_predictions_function},
      calculate_custom_differentiate_predictions_wrt_linear_predictor_function{calculate_custom_differentiate_predictions_wrt_linear_predictor_function},
      boosting_steps_before_pruning_is_done{boosting_steps_before_pruning_is_done}, boosting_steps_before_interactions_are_allowed{boosting_steps_before_interactions_are_allowed}
{
}

APLRRegressor::APLRRegressor(const APLRRegressor &other)
    : reserved_terms_times_num_x{other.reserved_terms_times_num_x}, intercept{other.intercept}, terms{other.terms}, m{other.m}, v{other.v},
      loss_function{other.loss_function}, link_function{other.link_function}, validation_ratio{other.validation_ratio},
      n_jobs{other.n_jobs}, random_state{other.random_state}, bins{other.bins},
      verbosity{other.verbosity}, term_names{other.term_names}, term_coefficients{other.term_coefficients},
      max_interaction_level{other.max_interaction_level}, intercept_steps{other.intercept_steps}, max_interactions{other.max_interactions},
      interactions_eligible{other.interactions_eligible}, validation_error_steps{other.validation_error_steps},
      min_observations_in_split{other.min_observations_in_split}, ineligible_boosting_steps_added{other.ineligible_boosting_steps_added},
      max_eligible_terms{other.max_eligible_terms}, number_of_base_terms{other.number_of_base_terms},
      feature_importance{other.feature_importance}, dispersion_parameter{other.dispersion_parameter}, min_training_prediction_or_response{other.min_training_prediction_or_response},
      max_training_prediction_or_response{other.max_training_prediction_or_response}, validation_tuning_metric{other.validation_tuning_metric},
      validation_indexes{other.validation_indexes}, quantile{other.quantile}, m_optimal{other.m_optimal},
      calculate_custom_validation_error_function{other.calculate_custom_validation_error_function},
      calculate_custom_loss_function{other.calculate_custom_loss_function}, calculate_custom_negative_gradient_function{other.calculate_custom_negative_gradient_function},
      calculate_custom_transform_linear_predictor_to_predictions_function{other.calculate_custom_transform_linear_predictor_to_predictions_function},
      calculate_custom_differentiate_predictions_wrt_linear_predictor_function{other.calculate_custom_differentiate_predictions_wrt_linear_predictor_function},
      boosting_steps_before_pruning_is_done{other.boosting_steps_before_pruning_is_done},
      boosting_steps_before_interactions_are_allowed{other.boosting_steps_before_interactions_are_allowed}
{
}

APLRRegressor::~APLRRegressor()
{
}

void APLRRegressor::fit(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight, const std::vector<std::string> &X_names,
                        const std::vector<size_t> &validation_set_indexes, const std::vector<size_t> &prioritized_predictors_indexes,
                        const std::vector<int> &monotonic_constraints, const VectorXi &group, const std::vector<std::vector<size_t>> &interaction_constraints,
                        const MatrixXd &other_data)
{
    throw_error_if_loss_function_does_not_exist();
    throw_error_if_link_function_does_not_exist();
    throw_error_if_dispersion_parameter_is_invalid();
    throw_error_if_m_is_invalid();
    validate_input_to_fit(X, y, sample_weight, X_names, validation_set_indexes, prioritized_predictors_indexes, monotonic_constraints, group,
                          interaction_constraints, other_data);
    define_training_and_validation_sets(X, y, sample_weight, validation_set_indexes, group, other_data);
    scale_training_observations_if_using_log_link_function();
    initialize(prioritized_predictors_indexes, monotonic_constraints, interaction_constraints);
    execute_boosting_steps();
    update_coefficients_for_all_steps();
    print_final_summary();
    find_optimal_m_and_update_model_accordingly();
    merge_similar_terms();
    remove_unused_terms();
    revert_scaling_if_using_log_link_function();
    name_terms(X, X_names);
    calculate_feature_importance_on_validation_set();
    find_min_and_max_training_predictions_or_responses();
    cleanup_after_fit();
}

void APLRRegressor::throw_error_if_loss_function_does_not_exist()
{
    bool loss_function_exists{false};
    if (loss_function == "mse")
        loss_function_exists = true;
    else if (loss_function == "binomial")
        loss_function_exists = true;
    else if (loss_function == "poisson")
        loss_function_exists = true;
    else if (loss_function == "gamma")
        loss_function_exists = true;
    else if (loss_function == "tweedie")
        loss_function_exists = true;
    else if (loss_function == "group_mse")
        loss_function_exists = true;
    else if (loss_function == "mae")
        loss_function_exists = true;
    else if (loss_function == "quantile")
        loss_function_exists = true;
    else if (loss_function == "negative_binomial")
        loss_function_exists = true;
    else if (loss_function == "cauchy")
        loss_function_exists = true;
    else if (loss_function == "weibull")
        loss_function_exists = true;
    else if (loss_function == "custom_function")
        loss_function_exists = true;
    if (!loss_function_exists)
        throw std::runtime_error("Loss function " + loss_function + " is not available in APLR.");
}

void APLRRegressor::throw_error_if_link_function_does_not_exist()
{
    bool link_function_exists{false};
    if (link_function == "identity")
        link_function_exists = true;
    else if (link_function == "logit")
        link_function_exists = true;
    else if (link_function == "log")
        link_function_exists = true;
    else if (link_function == "custom_function")
        link_function_exists = true;
    if (!link_function_exists)
        throw std::runtime_error("Link function " + link_function + " is not available in APLR.");
}

void APLRRegressor::throw_error_if_dispersion_parameter_is_invalid()
{
    if (loss_function == "tweedie")
    {
        bool dispersion_parameter_equals_invalid_poits{is_approximately_equal(dispersion_parameter, 1.0) || is_approximately_equal(dispersion_parameter, 2.0)};
        bool dispersion_parameter_is_in_invalid_range{std::isless(dispersion_parameter, 1.0)};
        bool dispersion_parameter_is_invalid{dispersion_parameter_equals_invalid_poits || dispersion_parameter_is_in_invalid_range};
        if (dispersion_parameter_is_invalid)
            throw std::runtime_error("Invalid dispersion_parameter (variance power). It must not equal 1.0 or 2.0 and cannot be below 1.0.");
    }
    else if (loss_function == "negative_binomial" || loss_function == "cauchy" || loss_function == "weibull")
    {
        bool dispersion_parameter_is_in_invalid{std::islessequal(dispersion_parameter, 0.0)};
        if (dispersion_parameter_is_in_invalid)
            throw std::runtime_error("Invalid dispersion_parameter. It must be greater than zero.");
    }
}

void APLRRegressor::throw_error_if_m_is_invalid()
{
    if (m < 1)
        throw std::runtime_error("The maximum number of boosting steps, m, must be at least 1.");
}

void APLRRegressor::validate_input_to_fit(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight,
                                          const std::vector<std::string> &X_names, const std::vector<size_t> &validation_set_indexes,
                                          const std::vector<size_t> &prioritized_predictors_indexes, const std::vector<int> &monotonic_constraints, const VectorXi &group,
                                          const std::vector<std::vector<size_t>> &interaction_constraints, const MatrixXd &other_data)
{
    if (X.rows() != y.size())
        throw std::runtime_error("X and y must have the same number of rows.");
    if (X.rows() < 2)
        throw std::runtime_error("X and y cannot have less than two rows.");
    if (X_names.size() > 0 && X_names.size() != static_cast<size_t>(X.cols()))
        throw std::runtime_error("X_names must have as many columns as X.");
    throw_error_if_matrix_has_nan_or_infinite_elements(X, "X");
    throw_error_if_matrix_has_nan_or_infinite_elements(y, "y");
    throw_error_if_matrix_has_nan_or_infinite_elements(sample_weight, "sample_weight");
    throw_error_if_validation_set_indexes_has_invalid_indexes(y, validation_set_indexes);
    throw_error_if_prioritized_predictors_indexes_has_invalid_indexes(X, prioritized_predictors_indexes);
    throw_error_if_monotonic_constraints_has_invalid_indexes(X, monotonic_constraints);
    throw_error_if_interaction_constraints_has_invalid_indexes(X, interaction_constraints);
    throw_error_if_response_contains_invalid_values(y);
    throw_error_if_sample_weight_contains_invalid_values(y, sample_weight);
    bool group_is_of_incorrect_size{loss_function == "group_mse" && group.rows() != y.rows()};
    if (group_is_of_incorrect_size)
        throw std::runtime_error("When loss_function is group_mse then y and group must have the same number of rows.");
    bool other_data_is_provided{other_data.size() > 0};
    if (other_data_is_provided)
    {
        bool other_data_is_of_incorrect_size{other_data.rows() != y.rows()};
        if (other_data_is_of_incorrect_size)
            throw std::runtime_error("other_data and y must have the same number of rows.");
    }
}

void APLRRegressor::throw_error_if_validation_set_indexes_has_invalid_indexes(const VectorXd &y, const std::vector<size_t> &validation_set_indexes)
{
    bool validation_set_indexes_is_provided{validation_set_indexes.size() > 0};
    if (validation_set_indexes_is_provided)
    {
        size_t max_index{*std::max_element(validation_set_indexes.begin(), validation_set_indexes.end())};
        bool validation_set_indexes_has_elements_out_of_bounds{max_index > static_cast<size_t>(y.size() - 1)};
        if (validation_set_indexes_has_elements_out_of_bounds)
            throw std::runtime_error("validation_set_indexes has elements that are out of bounds.");
    }
}

void APLRRegressor::throw_error_if_prioritized_predictors_indexes_has_invalid_indexes(const MatrixXd &X, const std::vector<size_t> &prioritized_predictors_indexes)
{
    bool prioritized_predictors_indexes_is_provided{prioritized_predictors_indexes.size() > 0};
    if (prioritized_predictors_indexes_is_provided)
    {
        size_t max_index{*std::max_element(prioritized_predictors_indexes.begin(), prioritized_predictors_indexes.end())};
        bool prioritized_predictors_indexes_has_elements_out_of_bounds{max_index > static_cast<size_t>(X.cols() - 1)};
        if (prioritized_predictors_indexes_has_elements_out_of_bounds)
            throw std::runtime_error("prioritized_predictors_indexes has elements that are out of bounds.");
    }
}

void APLRRegressor::throw_error_if_monotonic_constraints_has_invalid_indexes(const MatrixXd &X, const std::vector<int> &monotonic_constraints)
{
    bool error{monotonic_constraints.size() > 0 && monotonic_constraints.size() != X.cols()};
    if (error)
        throw std::runtime_error("monotonic_constraints must either be empty or a vector with one integer for each column in X.");
}

void APLRRegressor::throw_error_if_interaction_constraints_has_invalid_indexes(const MatrixXd &X, const std::vector<std::vector<size_t>> &interaction_constraints)
{
    for (auto &legal_interaction_combination : interaction_constraints)
    {
        bool illegal_size_of_legal_combination{legal_interaction_combination.size() == 0};
        if (illegal_size_of_legal_combination)
            throw std::runtime_error("At least one entry in interaction_constraints is empty. Please remove empty entries.");
        for (auto &predictor_index : legal_interaction_combination)
        {
            bool illegal_predictor_index{predictor_index > X.cols()};
            if (illegal_predictor_index)
                throw std::runtime_error("Illegal predictor index " + std::to_string(predictor_index) + " found in interaction_constraints.");
        }
    }
}

void APLRRegressor::throw_error_if_response_contains_invalid_values(const VectorXd &y)
{
    if (link_function == "logit" || loss_function == "binomial")
    {
        std::string error_message{"Response values for the logit link function or binomial loss_function cannot be less than zero or greater than one."};
        throw_error_if_response_is_not_between_0_and_1(y, error_message);
    }
    else if (loss_function == "gamma" || (loss_function == "tweedie" && std::isgreater(dispersion_parameter, 2)))
    {
        std::string error_message;
        if (loss_function == "tweedie")
            error_message = "Response values for the " + loss_function + " loss_function when dispersion_parameter>2 must be greater than zero.";
        else
            error_message = "Response values for the " + loss_function + " loss_function must be greater than zero.";
        throw_error_if_response_is_not_greater_than_zero(y, error_message);
    }
    else if (link_function == "log" || loss_function == "poisson" || loss_function == "negative_binomial" || loss_function == "weibull" || (loss_function == "tweedie" && std::isless(dispersion_parameter, 2) && std::isgreater(dispersion_parameter, 1)))
    {
        std::string error_message{"Response values for the log link function or poisson loss_function or negative binomial loss function or weibull loss function or tweedie loss_function when dispersion_parameter<2 cannot be less than zero."};
        throw_error_if_vector_contains_negative_values(y, error_message);
    }
    else if (validation_tuning_metric == "negative_gini")
    {
        std::string error_message{"Response values cannot be negative when using the negative_gini validation_tuning_metric."};
        throw_error_if_vector_contains_negative_values(y, error_message);
        bool sum_is_zero{y.sum() == 0};
        if (sum_is_zero)
            throw std::runtime_error("Response values cannot sum to zero when using the negative_gini validation_tuning_metric.");
    }
}

void APLRRegressor::throw_error_if_response_is_not_between_0_and_1(const VectorXd &y, const std::string &error_message)
{
    bool response_is_less_than_zero{(y.array() < 0.0).any()};
    bool response_is_greater_than_one{(y.array() > 1.0).any()};
    if (response_is_less_than_zero || response_is_greater_than_one)
        throw std::runtime_error(error_message);
}

void APLRRegressor::throw_error_if_vector_contains_negative_values(const VectorXd &y, const std::string &error_message)
{
    bool vector_is_less_than_zero{(y.array() < 0.0).any()};
    if (vector_is_less_than_zero)
        throw std::runtime_error(error_message);
}

void APLRRegressor::throw_error_if_response_is_not_greater_than_zero(const VectorXd &y, const std::string &error_message)
{
    bool response_is_not_greater_than_zero{(y.array() <= 0.0).any()};
    if (response_is_not_greater_than_zero)
        throw std::runtime_error(error_message);
}

void APLRRegressor::throw_error_if_sample_weight_contains_invalid_values(const VectorXd &y, const VectorXd &sample_weight)
{
    bool sample_weight_are_provided{sample_weight.size() > 0};
    if (sample_weight_are_provided)
    {
        if (sample_weight.size() != y.size())
            throw std::runtime_error("sample_weight must have 0 or as many rows as X and y.");
        throw_error_if_vector_contains_negative_values(sample_weight, "sample_weight cannot contain negative values.");
        bool sum_is_zero{sample_weight.sum() == 0};
        if (sum_is_zero)
            throw std::runtime_error("sample_weight cannot sum to zero.");
    }
}

void APLRRegressor::define_training_and_validation_sets(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight,
                                                        const std::vector<size_t> &validation_set_indexes, const VectorXi &group,
                                                        const MatrixXd &other_data)
{
    size_t y_size{static_cast<size_t>(y.size())};
    std::vector<size_t> train_indexes;
    bool use_validation_set_indexes{validation_set_indexes.size() > 0};
    if (use_validation_set_indexes)
    {
        std::vector<size_t> all_indexes(y_size);
        std::iota(std::begin(all_indexes), std::end(all_indexes), 0);
        validation_indexes = validation_set_indexes;
        train_indexes.reserve(y_size - validation_indexes.size());
        std::remove_copy_if(all_indexes.begin(), all_indexes.end(), std::back_inserter(train_indexes), [this](const size_t &arg)
                            { return (std::find(validation_indexes.begin(), validation_indexes.end(), arg) != validation_indexes.end()); });
    }
    else
    {
        train_indexes.reserve(y_size);
        validation_indexes = std::vector<size_t>(0);
        validation_indexes.reserve(y_size);
        std::mt19937 mersenne{random_state};
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double roll;
        for (size_t i = 0; i < y_size; ++i)
        {
            roll = distribution(mersenne);
            bool place_in_validation_set{std::isless(roll, validation_ratio)};
            if (place_in_validation_set)
            {
                validation_indexes.push_back(i);
            }
            else
            {
                train_indexes.push_back(i);
            }
        }
        train_indexes.shrink_to_fit();
        validation_indexes.shrink_to_fit();
    }
    // Defining train and test matrices
    X_train.resize(train_indexes.size(), X.cols());
    y_train.resize(train_indexes.size());
    sample_weight_train.resize(std::min(train_indexes.size(), static_cast<size_t>(sample_weight.size())));
    X_validation.resize(validation_indexes.size(), X.cols());
    y_validation.resize(validation_indexes.size());
    sample_weight_validation.resize(std::min(validation_indexes.size(), static_cast<size_t>(sample_weight.size())));
    // Populating train matrices
    for (size_t i = 0; i < train_indexes.size(); ++i)
    {
        X_train.row(i) = X.row(train_indexes[i]);
        y_train[i] = y[train_indexes[i]];
    }
    bool sample_weight_exist{sample_weight_train.size() == y_train.size()};
    if (sample_weight_exist)
    {
        for (size_t i = 0; i < train_indexes.size(); ++i)
        {
            sample_weight_train[i] = sample_weight[train_indexes[i]];
        }
    }
    bool groups_are_provided{group.size() > 0};
    if (groups_are_provided)
    {
        group_train.resize(train_indexes.size());
        for (size_t i = 0; i < train_indexes.size(); ++i)
        {
            group_train[i] = group[train_indexes[i]];
        }
        unique_groups_train = get_unique_integers(group_train);
    }
    bool other_data_is_provided{other_data.size() > 0};
    if (other_data_is_provided)
    {
        other_data_train.resize(train_indexes.size(), other_data.cols());
        for (size_t i = 0; i < train_indexes.size(); ++i)
        {
            other_data_train.row(i) = other_data.row(train_indexes[i]);
        }
    }
    // Populating test matrices
    for (size_t i = 0; i < validation_indexes.size(); ++i)
    {
        X_validation.row(i) = X.row(validation_indexes[i]);
        y_validation[i] = y[validation_indexes[i]];
    }
    sample_weight_exist = sample_weight_validation.size() == y_validation.size();
    if (sample_weight_exist)
    {
        for (size_t i = 0; i < validation_indexes.size(); ++i)
        {
            sample_weight_validation[i] = sample_weight[validation_indexes[i]];
        }
    }
    if (groups_are_provided)
    {
        group_validation.resize(validation_indexes.size());
        for (size_t i = 0; i < validation_indexes.size(); ++i)
        {
            group_validation[i] = group[validation_indexes[i]];
        }
        unique_groups_validation = get_unique_integers(group_validation);
    }
    if (other_data_is_provided)
    {
        other_data_validation.resize(validation_indexes.size(), other_data.cols());
        for (size_t i = 0; i < validation_indexes.size(); ++i)
        {
            other_data_validation.row(i) = other_data.row(validation_indexes[i]);
        }
    }
}

void APLRRegressor::scale_training_observations_if_using_log_link_function()
{
    if (link_function == "log")
    {
        double inverse_scaling_factor{y_train.maxCoeff() / std::exp(1)};
        bool inverse_scaling_factor_is_not_zero{!is_approximately_zero(inverse_scaling_factor)};
        if (inverse_scaling_factor_is_not_zero)
        {
            scaling_factor_for_log_link_function = 1 / inverse_scaling_factor;
            y_train *= scaling_factor_for_log_link_function;
        }
        else
            scaling_factor_for_log_link_function = 1.0;
    }
}

void APLRRegressor::initialize(const std::vector<size_t> &prioritized_predictors_indexes, const std::vector<int> &monotonic_constraints,
                               const std::vector<std::vector<size_t>> &interaction_constraints)
{
    number_of_base_terms = static_cast<size_t>(X_train.cols());

    terms.clear();
    terms.reserve(X_train.cols() * reserved_terms_times_num_x);

    intercept = 0;
    intercept_steps = VectorXd::Constant(m, 0);

    terms_eligible_current.reserve(X_train.cols() * reserved_terms_times_num_x);
    size_t X_train_cols{static_cast<size_t>(X_train.cols())};
    for (size_t i = 0; i < X_train_cols; ++i)
    {
        bool term_has_one_unique_value{check_if_base_term_has_only_one_unique_value(i)};
        Term copy_of_base_term{Term(i)};
        add_term_to_terms_eligible_current(copy_of_base_term);
        if (term_has_one_unique_value)
        {
            terms_eligible_current[terms_eligible_current.size() - 1].ineligible_boosting_steps = std::numeric_limits<size_t>::max();
        }
    }

    predictor_indexes.resize(X_train.cols());
    for (size_t i = 0; i < X_train_cols; ++i)
    {
        predictor_indexes[i] = i;
    }
    this->prioritized_predictors_indexes = prioritized_predictors_indexes;

    this->monotonic_constraints = monotonic_constraints;
    bool monotonic_constraints_provided{monotonic_constraints.size() > 0};
    if (monotonic_constraints_provided)
    {
        for (auto &term_eligible_current : terms_eligible_current)
        {
            term_eligible_current.set_monotonic_constraint(monotonic_constraints[term_eligible_current.base_term]);
        }
    }

    this->interaction_constraints = interaction_constraints;
    for (auto &legal_interaction_combination : this->interaction_constraints)
    {
        legal_interaction_combination = remove_duplicate_elements_from_vector(legal_interaction_combination);
    }

    linear_predictor_current = VectorXd::Constant(y_train.size(), intercept);
    linear_predictor_null_model = linear_predictor_current;
    linear_predictor_current_validation = VectorXd::Constant(y_validation.size(), intercept);
    predictions_current = transform_linear_predictor_to_predictions(linear_predictor_current, link_function, calculate_custom_transform_linear_predictor_to_predictions_function);
    predictions_current_validation = transform_linear_predictor_to_predictions(linear_predictor_current_validation, link_function, calculate_custom_transform_linear_predictor_to_predictions_function);

    validation_error_steps.resize(m);
    validation_error_steps.setConstant(std::numeric_limits<double>::infinity());

    update_gradient_and_errors();
}

bool APLRRegressor::check_if_base_term_has_only_one_unique_value(size_t base_term)
{
    size_t rows{static_cast<size_t>(X_train.rows())};
    if (rows == 1)
        return true;

    bool term_has_one_unique_value{true};
    for (size_t i = 1; i < rows; ++i)
    {
        bool observation_is_equal_to_previous{is_approximately_equal(X_train.col(base_term)[i], X_train.col(base_term)[i - 1])};
        if (!observation_is_equal_to_previous)
        {
            term_has_one_unique_value = false;
            break;
        }
    }

    return term_has_one_unique_value;
}

void APLRRegressor::add_term_to_terms_eligible_current(Term &term)
{
    terms_eligible_current.push_back(term);
}

VectorXd APLRRegressor::calculate_neg_gradient_current(const VectorXd &sample_weight_train)
{
    VectorXd output;
    if (loss_function == "mse")
        output = y_train - predictions_current;
    else if (loss_function == "binomial")
        output = y_train.array() / predictions_current.array() - (y_train.array() - 1.0) / (predictions_current.array() - 1.0);
    else if (loss_function == "poisson")
        output = y_train.array() / predictions_current.array() - 1;
    else if (loss_function == "gamma")
        output = (y_train.array() - predictions_current.array()) / predictions_current.array() / predictions_current.array();
    else if (loss_function == "tweedie")
        output = (y_train.array() - predictions_current.array()).array() * predictions_current.array().pow(-dispersion_parameter);
    else if (loss_function == "group_mse")
    {
        GroupData group_residuals_and_count{calculate_group_errors_and_count(y_train, predictions_current, group_train, unique_groups_train)};

        for (int unique_group_value : unique_groups_train)
        {
            group_residuals_and_count.error[unique_group_value] /= group_residuals_and_count.count[unique_group_value];
        }

        output = VectorXd(y_train.rows());
        for (Eigen::Index i = 0; i < y_train.size(); ++i)
        {
            output[i] = group_residuals_and_count.error[group_train[i]];
        }
    }
    else if (loss_function == "mae")
    {
        double mae{calculate_errors(y_train, predictions_current, sample_weight_train, "mae").mean()};
        output = (y_train.array() - predictions_current.array()).sign() * mae;
    }
    else if (loss_function == "quantile")
    {
        double mae{calculate_errors(y_train, predictions_current, sample_weight_train, "mae").mean()};
        output = (y_train.array() - predictions_current.array()).sign() * mae;
        for (Eigen::Index i = 0; i < y_train.size(); ++i)
        {
            if (y_train[i] < predictions_current[i])
                output[i] *= 1 - quantile;
            else
                output[i] *= quantile;
        }
    }
    else if (loss_function == "negative_binomial")
    {
        output = (y_train.array() - predictions_current.array()) / (predictions_current.array() * (dispersion_parameter * predictions_current.array() + 1));
    }
    else if (loss_function == "cauchy")
    {
        ArrayXd residuals{y_train.array() - predictions_current.array()};
        output = 2 * residuals / (dispersion_parameter * dispersion_parameter + residuals.pow(2));
    }
    else if (loss_function == "weibull")
    {
        output = dispersion_parameter / predictions_current.array() * ((y_train.array() / predictions_current.array()).pow(dispersion_parameter) - 1);
    }
    else if (loss_function == "custom_function")
    {
        try
        {
            output = calculate_custom_negative_gradient_function(y_train, predictions_current, group_train, other_data_train);
        }
        catch (const std::exception &e)
        {
            std::string error_msg{"Error when calculating custom negative gradient function: " + static_cast<std::string>(e.what())};
            throw std::runtime_error(error_msg);
        }
    }

    if (link_function != "identity")
        output = output.array() * differentiate_predictions_wrt_linear_predictor().array();

    return output;
}

VectorXd APLRRegressor::differentiate_predictions_wrt_linear_predictor()
{
    if (link_function == "logit")
        return 1.0 / 4.0 * (linear_predictor_current.array() / 2.0).cosh().array().pow(-2);
    else if (link_function == "log")
    {
        return linear_predictor_current.array().exp();
    }
    else if (link_function == "custom_function")
    {
        try
        {
            return calculate_custom_differentiate_predictions_wrt_linear_predictor_function(linear_predictor_current);
        }
        catch (const std::exception &e)
        {
            std::string error_msg{"Error when executing calculate_custom_differentiate_predictions_wrt_linear_predictor_function: " + static_cast<std::string>(e.what())};
            throw std::runtime_error(error_msg);
        }
    }
    return VectorXd(0);
}

void APLRRegressor::execute_boosting_steps()
{
    abort_boosting = false;
    for (size_t boosting_step = 0; boosting_step < m; ++boosting_step)
    {
        execute_boosting_step(boosting_step);
        if (abort_boosting)
            break;
    }
}

void APLRRegressor::execute_boosting_step(size_t boosting_step)
{
    update_intercept(boosting_step);
    bool prioritize_predictors{!abort_boosting && prioritized_predictors_indexes.size() > 0};
    if (prioritize_predictors)
    {
        for (auto &index : prioritized_predictors_indexes)
        {
            std::vector<size_t> terms_eligible_current_indexes_for_a_base_term{find_terms_eligible_current_indexes_for_a_base_term(index)};
            bool eligible_terms_exist{terms_eligible_current_indexes_for_a_base_term.size() > 0};
            if (eligible_terms_exist)
            {
                estimate_split_point_for_each_term(terms_eligible_current, terms_eligible_current_indexes_for_a_base_term);
                best_term_index = find_best_term_index(terms_eligible_current, terms_eligible_current_indexes_for_a_base_term);
                std::vector<size_t> predictor_index{index};
                consider_interactions(predictor_index, boosting_step);
                select_the_best_term_and_update_errors(boosting_step);
            }
        }
    }
    if (!abort_boosting)
    {
        std::vector<size_t> term_indexes{create_term_indexes(terms_eligible_current)};
        estimate_split_point_for_each_term(terms_eligible_current, term_indexes);
        best_term_index = find_best_term_index(terms_eligible_current, term_indexes);
        consider_interactions(predictor_indexes, boosting_step);
        select_the_best_term_and_update_errors(boosting_step);
        prune_terms(boosting_step);
    }
    if (abort_boosting)
        return;
    update_term_eligibility();
    print_summary_after_boosting_step(boosting_step);
}

void APLRRegressor::update_intercept(size_t boosting_step)
{
    double intercept_update;
    if (sample_weight_train.size() == 0)
        intercept_update = v * neg_gradient_current.mean();
    else
        intercept_update = v * (neg_gradient_current.array() * sample_weight_train.array()).sum() / sample_weight_train.array().sum();
    linear_predictor_update = VectorXd::Constant(neg_gradient_current.size(), intercept_update);
    linear_predictor_update_validation = VectorXd::Constant(y_validation.size(), intercept_update);
    update_linear_predictor_and_predictions();
    update_gradient_and_errors();
    calculate_and_validate_validation_error(boosting_step);
    if (!abort_boosting)
    {
        intercept += intercept_update;
        intercept_steps[boosting_step] = intercept;
    }
}

void APLRRegressor::update_linear_predictor_and_predictions()
{
    linear_predictor_current += linear_predictor_update;
    linear_predictor_current_validation += linear_predictor_update_validation;
    predictions_current = transform_linear_predictor_to_predictions(linear_predictor_current, link_function, calculate_custom_transform_linear_predictor_to_predictions_function);
    predictions_current_validation = transform_linear_predictor_to_predictions(linear_predictor_current_validation, link_function, calculate_custom_transform_linear_predictor_to_predictions_function);
}

void APLRRegressor::update_gradient_and_errors()
{
    neg_gradient_current = calculate_neg_gradient_current(sample_weight_train);
    neg_gradient_nullmodel_errors_sum = calculate_sum_error(calculate_errors(neg_gradient_current, linear_predictor_null_model, sample_weight_train, MSE_LOSS_FUNCTION));
}

std::vector<size_t> APLRRegressor::find_terms_eligible_current_indexes_for_a_base_term(size_t base_term)
{
    std::vector<size_t> terms_eligible_current_indexes_for_a_base_term;
    terms_eligible_current_indexes_for_a_base_term.reserve(terms_eligible_current.size());
    for (size_t i = 0; i < terms_eligible_current.size(); ++i)
    {
        bool term_is_eligible{terms_eligible_current[i].base_term == base_term && terms_eligible_current[i].ineligible_boosting_steps == 0};
        if (term_is_eligible)
            terms_eligible_current_indexes_for_a_base_term.push_back(i);
    }
    terms_eligible_current_indexes_for_a_base_term.shrink_to_fit();
    return terms_eligible_current_indexes_for_a_base_term;
}

void APLRRegressor::estimate_split_point_for_each_term(std::vector<Term> &terms, std::vector<size_t> &terms_indexes)
{
    bool multithreading{n_jobs != 1 && terms_indexes.size() > 1};
    if (multithreading)
    {
        distributed_terms = distribute_terms_indexes_to_cores(terms_indexes, n_jobs);

        std::vector<std::thread> threads(distributed_terms.size());

        auto estimate_split_point_for_distributed_terms_in_one_thread = [this, &terms, &terms_indexes](size_t thread_index)
        {
            for (size_t i = 0; i < distributed_terms[thread_index].size(); ++i)
            {
                terms[terms_indexes[distributed_terms[thread_index][i]]].estimate_split_point(X_train, neg_gradient_current, sample_weight_train, bins, v, min_observations_in_split);
            }
        };

        for (size_t i = 0; i < threads.size(); ++i)
        {
            threads[i] = std::thread(estimate_split_point_for_distributed_terms_in_one_thread, i);
        }
        for (size_t i = 0; i < threads.size(); ++i)
        {
            threads[i].join();
        }
    }
    else
    {
        for (size_t i = 0; i < terms_indexes.size(); ++i)
        {
            terms[terms_indexes[i]].estimate_split_point(X_train, neg_gradient_current, sample_weight_train, bins, v, min_observations_in_split);
        }
    }
}

size_t APLRRegressor::find_best_term_index(std::vector<Term> &terms, std::vector<size_t> &terms_indexes)
{
    size_t best_term_index{std::numeric_limits<size_t>::max()};
    double lowest_errors_sum{neg_gradient_nullmodel_errors_sum};

    for (auto &term_index : terms_indexes)
    {
        bool term_is_eligible{terms[term_index].ineligible_boosting_steps == 0};
        if (term_is_eligible)
        {
            if (std::isless(terms[term_index].split_point_search_errors_sum, lowest_errors_sum))
            {
                best_term_index = term_index;
                lowest_errors_sum = terms[term_index].split_point_search_errors_sum;
            }
        }
    }

    return best_term_index;
}

void APLRRegressor::consider_interactions(const std::vector<size_t> &available_predictor_indexes, size_t boosting_step)
{
    bool consider_interactions{terms.size() > 0 && max_interaction_level > 0 && interactions_eligible < max_interactions && boosting_step >= boosting_steps_before_interactions_are_allowed};
    if (consider_interactions)
    {
        determine_interactions_to_consider(available_predictor_indexes);
        std::vector<size_t> interactions_to_consider_indexes{create_term_indexes(interactions_to_consider)};
        estimate_split_point_for_each_term(interactions_to_consider, interactions_to_consider_indexes);
        find_sorted_indexes_for_errors_for_interactions_to_consider();
        add_promising_interactions_and_select_the_best_one();
    }
}

void APLRRegressor::determine_interactions_to_consider(const std::vector<size_t> &available_predictor_indexes)
{
    interactions_to_consider = std::vector<Term>();
    interactions_to_consider.reserve(static_cast<size_t>(X_train.cols()) * terms.size());
    bool monotonic_constraints_provided{monotonic_constraints.size() > 0};
    bool interaction_constraints_provided{interaction_constraints.size() > 0};

    VectorXi indexes_for_terms_to_consider_as_interaction_partners{find_indexes_for_terms_to_consider_as_interaction_partners()};
    for (auto &model_term_index : indexes_for_terms_to_consider_as_interaction_partners)
    {
        for (auto &new_term_index : available_predictor_indexes)
        {
            bool term_is_eligible{terms_eligible_current[new_term_index].ineligible_boosting_steps == 0};
            if (term_is_eligible)
            {
                Term interaction{Term(new_term_index)};
                if (monotonic_constraints_provided)
                    interaction.set_monotonic_constraint(monotonic_constraints[new_term_index]);
                Term model_term_without_given_terms{terms[model_term_index]};
                model_term_without_given_terms.given_terms.clear();
                model_term_without_given_terms.cleanup_when_this_term_was_added_as_a_given_term();
                Term model_term_with_added_given_term{terms[model_term_index]};
                bool model_term_without_given_terms_can_be_a_given_term{model_term_without_given_terms.get_monotonic_constraint() == 0};
                if (model_term_without_given_terms_can_be_a_given_term)
                    model_term_with_added_given_term.given_terms.push_back(model_term_without_given_terms);
                add_necessary_given_terms_to_interaction(interaction, model_term_with_added_given_term);
                if (interaction_constraints_provided)
                {
                    bool interaction_violates_constraints{true};
                    bool interaction_found_in_at_least_one_interaction_combination{false};
                    for (auto &legal_interaction_combination : interaction_constraints)
                    {
                        InteractionConstraintsTest interaction_constraints_test{interaction.test_interaction_constraints(legal_interaction_combination)};
                        if (interaction_constraints_test.at_least_one_term_found_in_combination)
                            interaction_found_in_at_least_one_interaction_combination = true;
                        if (interaction_constraints_test.term_adheres_to_combination)
                            interaction_violates_constraints = false;
                    }
                    interaction_violates_constraints = interaction_found_in_at_least_one_interaction_combination && interaction_violates_constraints;
                    if (interaction_violates_constraints)
                        continue;
                }
                bool interaction_is_invalid{interaction.given_terms.size() == 0};
                if (interaction_is_invalid)
                    continue;
                bool interaction_level_is_too_high{interaction.get_interaction_level() > max_interaction_level};
                if (interaction_level_is_too_high)
                    continue;
                bool interaction_is_already_in_the_model{false};
                for (auto &term : terms)
                {
                    if (interaction == term)
                    {
                        interaction_is_already_in_the_model = true;
                        break;
                    }
                }
                if (interaction_is_already_in_the_model)
                    continue;
                bool interaction_already_exists_in_terms_eligible_current{false};
                for (auto &term_eligible_current : terms_eligible_current)
                {
                    if (interaction.base_term == term_eligible_current.base_term && Term::equals_given_terms(interaction, term_eligible_current))
                    {
                        interaction_already_exists_in_terms_eligible_current = true;
                        break;
                    }
                }
                if (interaction_already_exists_in_terms_eligible_current)
                    continue;
                interactions_to_consider.push_back(interaction);
            }
        }
    }
}

VectorXi APLRRegressor::find_indexes_for_terms_to_consider_as_interaction_partners()
{
    size_t number_of_terms_to_consider_as_interaction_partners{find_out_how_many_terms_to_consider_as_interaction_partners()};
    VectorXd split_point_errors(terms.size());
    VectorXi indexes_for_terms_to_consider_as_interaction_partners(terms.size());
    size_t count{0};
    for (size_t i = 0; i < terms.size(); ++i)
    {
        if (terms[i].get_can_be_used_as_a_given_term())
        {
            split_point_errors[count] = terms[i].split_point_search_errors_sum;
            indexes_for_terms_to_consider_as_interaction_partners[count] = i;
            ++count;
        }
    }
    split_point_errors.conservativeResize(count);
    indexes_for_terms_to_consider_as_interaction_partners.conservativeResize(count);
    bool selecting_the_terms_with_lowest_previous_errors_is_necessary{number_of_terms_to_consider_as_interaction_partners < indexes_for_terms_to_consider_as_interaction_partners.size()};
    if (selecting_the_terms_with_lowest_previous_errors_is_necessary)
    {
        VectorXi sorted_indexes{sort_indexes_ascending(split_point_errors)};
        VectorXi temp_indexes(number_of_terms_to_consider_as_interaction_partners);
        for (size_t i = 0; i < number_of_terms_to_consider_as_interaction_partners; ++i)
        {
            temp_indexes[i] = indexes_for_terms_to_consider_as_interaction_partners[sorted_indexes[i]];
        }
        indexes_for_terms_to_consider_as_interaction_partners = std::move(temp_indexes);
    }
    return indexes_for_terms_to_consider_as_interaction_partners;
}

size_t APLRRegressor::find_out_how_many_terms_to_consider_as_interaction_partners()
{
    size_t terms_to_consider{terms.size()};
    if (max_eligible_terms > 0)
    {
        terms_to_consider = std::min(max_eligible_terms, terms.size());
    }
    return terms_to_consider;
}

void APLRRegressor::add_necessary_given_terms_to_interaction(Term &interaction, Term &existing_model_term)
{
    bool model_term_has_multiple_given_terms{existing_model_term.given_terms.size() > 1};
    if (model_term_has_multiple_given_terms)
    {
        MatrixXi value_indicator_for_each_given_term(X_train.rows(), existing_model_term.given_terms.size());
        for (size_t col = 0; col < existing_model_term.given_terms.size(); ++col)
        {
            value_indicator_for_each_given_term.col(col) = calculate_indicator(existing_model_term.given_terms[col].calculate(X_train));
        }
        for (size_t col = 0; col < existing_model_term.given_terms.size(); ++col)
        {
            VectorXi combined_value_indicator_for_the_other_given_terms{VectorXi::Constant(X_train.rows(), 1)};
            for (size_t col2 = 0; col2 < existing_model_term.given_terms.size(); ++col2)
            {
                bool is_other_given_term{col2 != col};
                if (is_other_given_term)
                {
                    combined_value_indicator_for_the_other_given_terms = combined_value_indicator_for_the_other_given_terms.array() * value_indicator_for_each_given_term.col(col2).array();
                }
            }

            bool given_term_provides_an_unique_zero{false};
            for (Eigen::Index row = 0; row < X_train.rows(); ++row)
            {
                given_term_provides_an_unique_zero = combined_value_indicator_for_the_other_given_terms[row] > 0 && value_indicator_for_each_given_term.col(col)[row] == 0;
                if (given_term_provides_an_unique_zero)
                    break;
            }

            if (given_term_provides_an_unique_zero)
                interaction.given_terms.push_back(existing_model_term.given_terms[col]);
        }
    }
    else
    {
        for (auto &given_term : existing_model_term.given_terms)
        {
            interaction.given_terms.push_back(given_term);
        }
    }
}

void APLRRegressor::find_sorted_indexes_for_errors_for_interactions_to_consider()
{
    VectorXd errors_for_interactions_to_consider(interactions_to_consider.size());
    for (size_t i = 0; i < interactions_to_consider.size(); ++i)
    {
        errors_for_interactions_to_consider[i] = interactions_to_consider[i].split_point_search_errors_sum;
    }
    sorted_indexes_of_errors_for_interactions_to_consider = sort_indexes_ascending(errors_for_interactions_to_consider);
}

void APLRRegressor::add_promising_interactions_and_select_the_best_one()
{
    size_t best_term_before_interactions{best_term_index};
    bool best_term_before_interactions_was_not_selected{best_term_before_interactions == std::numeric_limits<size_t>::max()};
    bool error_is_less_than_for_best_term_before_interactions;
    for (Eigen::Index j = 0; j < sorted_indexes_of_errors_for_interactions_to_consider.size(); ++j) // For each interaction to consider starting from lowest to highest error
    {
        bool allowed_to_add_one_interaction{interactions_eligible < max_interactions};
        if (allowed_to_add_one_interaction)
        {
            if (best_term_before_interactions_was_not_selected)
                error_is_less_than_for_best_term_before_interactions = std::isless(interactions_to_consider[sorted_indexes_of_errors_for_interactions_to_consider[j]].split_point_search_errors_sum, neg_gradient_nullmodel_errors_sum);
            else
                error_is_less_than_for_best_term_before_interactions = std::isless(interactions_to_consider[sorted_indexes_of_errors_for_interactions_to_consider[j]].split_point_search_errors_sum, terms_eligible_current[best_term_before_interactions].split_point_search_errors_sum);

            if (error_is_less_than_for_best_term_before_interactions)
            {
                add_term_to_terms_eligible_current(interactions_to_consider[sorted_indexes_of_errors_for_interactions_to_consider[j]]);
                bool is_best_interaction{j == 0};
                if (is_best_interaction)
                    best_term_index = terms_eligible_current.size() - 1;
                ++interactions_eligible;
            }
            else
                break;
        }
    }
}

void APLRRegressor::select_the_best_term_and_update_errors(size_t boosting_step)
{
    bool no_term_was_selected{best_term_index == std::numeric_limits<size_t>::max()};
    if (no_term_was_selected)
        return;

    linear_predictor_update = terms_eligible_current[best_term_index].calculate_contribution_to_linear_predictor(X_train);
    linear_predictor_update_validation = terms_eligible_current[best_term_index].calculate_contribution_to_linear_predictor(X_validation);
    update_linear_predictor_and_predictions();
    update_gradient_and_errors();
    double backup_of_validation_error{validation_error_steps[boosting_step]};
    calculate_and_validate_validation_error(boosting_step);
    if (abort_boosting)
        validation_error_steps[boosting_step] = backup_of_validation_error;
    else
        update_terms(boosting_step);
}

void APLRRegressor::remove_ineligibility()
{
    for (auto &term : terms_eligible_current)
    {
        bool make_eligible{term.ineligible_boosting_steps > 0 && term.ineligible_boosting_steps < std::numeric_limits<size_t>::max()};
        if (make_eligible)
        {
            term.ineligible_boosting_steps = 0;
        }
    }
}

void APLRRegressor::update_terms(size_t boosting_step)
{
    bool no_term_is_in_model{terms.size() == 0};
    if (no_term_is_in_model)
        add_new_term(boosting_step);
    else
    {
        bool found{false};
        for (size_t j = 0; j < terms.size(); ++j)
        {
            bool term_is_already_in_model{terms[j] == terms_eligible_current[best_term_index]};
            if (term_is_already_in_model)
            {
                terms[j].coefficient += terms_eligible_current[best_term_index].coefficient;
                terms[j].coefficient_steps[boosting_step] = terms[j].coefficient;
                found = true;
                break;
            }
        }
        if (!found)
        {
            add_new_term(boosting_step);
        }
    }
}

void APLRRegressor::add_new_term(size_t boosting_step)
{
    terms_eligible_current[best_term_index].coefficient_steps = VectorXd::Constant(m, 0);
    terms_eligible_current[best_term_index].coefficient_steps[boosting_step] = terms_eligible_current[best_term_index].coefficient;
    terms.push_back(Term(terms_eligible_current[best_term_index]));
}

void APLRRegressor::prune_terms(size_t boosting_step)
{
    bool prune{boosting_steps_before_pruning_is_done > 0 && (boosting_step + 1) % boosting_steps_before_pruning_is_done == 0 && boosting_step > 0};
    if (!prune)
    {
        pruning_was_done_in_the_current_boosting_step = false;
        return;
    }

    pruning_was_done_in_the_current_boosting_step = true;
    double best_error{neg_gradient_nullmodel_errors_sum};
    double new_error;
    size_t index_for_term_to_remove{std::numeric_limits<size_t>::max()};
    size_t terms_pruned{0};
    do
    {
        new_error = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < terms.size(); ++i)
        {
            bool term_is_used{!is_approximately_zero(terms[i].coefficient)};
            if (term_is_used)
            {
                linear_predictor_update = -terms[i].calculate_contribution_to_linear_predictor(X_train);
                double error_after_pruning_term = calculate_sum_error(calculate_errors(neg_gradient_current, linear_predictor_update, sample_weight_train, MSE_LOSS_FUNCTION));
                bool improvement{std::islessequal(error_after_pruning_term, new_error)};
                if (improvement)
                {
                    new_error = error_after_pruning_term;
                    index_for_term_to_remove = i;
                }
            }
        }
        bool removal_of_term_is_better{std::islessequal(new_error, best_error)};
        if (removal_of_term_is_better)
        {
            linear_predictor_update = -terms[index_for_term_to_remove].calculate_contribution_to_linear_predictor(X_train);
            terms[index_for_term_to_remove].coefficient = 0.0;
            update_linear_predictor_and_predictions();
            update_gradient_and_errors();
            update_intercept(boosting_step);
            new_error = neg_gradient_nullmodel_errors_sum;
            best_error = new_error;
            ++terms_pruned;
        }
    } while (std::islessequal(new_error, best_error) && terms_pruned < terms.size());
    if (terms_pruned > 0)
    {
        remove_unused_terms();
        remove_ineligibility();
    }
}

void APLRRegressor::calculate_and_validate_validation_error(size_t boosting_step)
{
    if (link_function == "log")
    {
        VectorXd rescaled_predictions_current_validation{predictions_current_validation / scaling_factor_for_log_link_function};
        calculate_validation_error(boosting_step, rescaled_predictions_current_validation);
    }
    else
        calculate_validation_error(boosting_step, predictions_current_validation);

    bool validation_error_is_invalid{!std::isfinite(validation_error_steps[boosting_step])};
    if (validation_error_is_invalid)
    {
        abort_boosting = true;
        std::string warning_message{"Warning: Encountered numerical problems when calculating validation error in the previous boosting step. Not continuing with further boosting steps. One potential reason is if the combination of loss_function and link_function is invalid."};
        std::cout << warning_message << "\n";
    }
}

void APLRRegressor::calculate_validation_error(size_t boosting_step, const VectorXd &predictions)
{
    if (validation_tuning_metric == "default")
    {
        if (loss_function == "custom_function")
        {
            try
            {
                validation_error_steps[boosting_step] = calculate_custom_loss_function(y_validation, predictions, sample_weight_validation, group_validation, other_data_validation);
            }
            catch (const std::exception &e)
            {
                std::string error_msg{"Error when calculating custom loss function: " + static_cast<std::string>(e.what())};
                throw std::runtime_error(error_msg);
            }
        }
        else
            validation_error_steps[boosting_step] = calculate_mean_error(calculate_errors(y_validation, predictions, sample_weight_validation, loss_function, dispersion_parameter, group_validation, unique_groups_validation, quantile), sample_weight_validation);
    }
    else if (validation_tuning_metric == "mse")
        validation_error_steps[boosting_step] = calculate_mean_error(calculate_errors(y_validation, predictions, sample_weight_validation, MSE_LOSS_FUNCTION), sample_weight_validation);
    else if (validation_tuning_metric == "mae")
        validation_error_steps[boosting_step] = calculate_mean_error(calculate_errors(y_validation, predictions, sample_weight_validation, "mae"), sample_weight_validation);
    else if (validation_tuning_metric == "negative_gini")
        validation_error_steps[boosting_step] = -calculate_gini(y_validation, predictions, sample_weight_validation);
    else if (validation_tuning_metric == "rankability")
        validation_error_steps[boosting_step] = -calculate_rankability(y_validation, predictions, sample_weight_validation, random_state);
    else if (validation_tuning_metric == "group_mse")
    {
        bool group_is_not_provided{group_validation.rows() == 0};
        if (group_is_not_provided)
            throw std::runtime_error("When validation_tuning_metric is group_mse then the group argument in fit() must be provided.");
        validation_error_steps[boosting_step] = calculate_mean_error(calculate_errors(y_validation, predictions, sample_weight_validation, "group_mse", dispersion_parameter, group_validation, unique_groups_validation, quantile), sample_weight_validation);
    }
    else if (validation_tuning_metric == "custom_function")
    {
        try
        {
            validation_error_steps[boosting_step] = calculate_custom_validation_error_function(y_validation, predictions, sample_weight_validation, group_validation, other_data_validation);
        }
        catch (const std::exception &e)
        {
            std::string error_msg{"Error when calculating custom validation error function: " + static_cast<std::string>(e.what())};
            throw std::runtime_error(error_msg);
        }
    }
    else
        throw std::runtime_error(validation_tuning_metric + " is an invalid validation_tuning_metric.");
}

void APLRRegressor::update_term_eligibility()
{
    bool eligibility_is_used{ineligible_boosting_steps_added > 0 && max_eligible_terms > 0 && !pruning_was_done_in_the_current_boosting_step};
    if (eligibility_is_used)
    {
        VectorXd terms_eligible_current_split_point_errors(terms_eligible_current.size());
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            terms_eligible_current_split_point_errors[i] = terms_eligible_current[i].split_point_search_errors_sum;
        }
        VectorXi sorted_split_point_errors_indexes{sort_indexes_ascending(terms_eligible_current_split_point_errors)};

        size_t count{0};
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            bool term_is_eligible_now{terms_eligible_current[sorted_split_point_errors_indexes[i]].ineligible_boosting_steps == 0};
            if (term_is_eligible_now)
            {
                ++count;
                bool term_should_receive_added_boosting_steps{count > max_eligible_terms};
                if (term_should_receive_added_boosting_steps)
                    terms_eligible_current[sorted_split_point_errors_indexes[i]].ineligible_boosting_steps += ineligible_boosting_steps_added;
            }
            else
            {
                terms_eligible_current[sorted_split_point_errors_indexes[i]].ineligible_boosting_steps -= 1;
            }
        }
    }
    number_of_eligible_terms = 0;
    for (size_t i = 0; i < terms_eligible_current.size(); ++i)
    {
        bool term_is_eligible{terms_eligible_current[i].ineligible_boosting_steps == 0};
        if (term_is_eligible)
            ++number_of_eligible_terms;
    }
}

void APLRRegressor::print_summary_after_boosting_step(size_t boosting_step)
{
    if (verbosity >= 2)
    {
        std::cout << "Boosting step: " << boosting_step + 1 << ". Unique terms: " << terms.size() << ". Terms eligible: " << number_of_eligible_terms << ". Validation error: " << validation_error_steps[boosting_step] << ".\n";
    }
}

void APLRRegressor::update_coefficients_for_all_steps()
{
    for (size_t j = 0; j < m; ++j)
    {
        bool fill_down_coefficient_steps{j > 0 && is_approximately_zero(intercept_steps[j]) && !is_approximately_zero(intercept_steps[j - 1])};
        if (fill_down_coefficient_steps)
            intercept_steps[j] = intercept_steps[j - 1];
    }

    for (size_t i = 0; i < terms.size(); ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            bool fill_down_coefficient_steps{j > 0 && is_approximately_zero(terms[i].coefficient_steps[j]) && !is_approximately_zero(terms[i].coefficient_steps[j - 1])};
            if (fill_down_coefficient_steps)
                terms[i].coefficient_steps[j] = terms[i].coefficient_steps[j - 1];
        }
    }
}

void APLRRegressor::print_final_summary()
{
    if (verbosity >= 1)
    {
        std::cout << "Unique terms: " << terms.size() << ". Terms available in final boosting step: " << terms_eligible_current.size() << ".\n";
    }
}

void APLRRegressor::find_optimal_m_and_update_model_accordingly()
{
    size_t best_boosting_step_index;
    validation_error_steps.minCoeff(&best_boosting_step_index);
    intercept = intercept_steps[best_boosting_step_index];
    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].coefficient = terms[i].coefficient_steps[best_boosting_step_index];
    }
    m_optimal = best_boosting_step_index + 1;
}

void APLRRegressor::merge_similar_terms()
{
    for (size_t i = 0; i < terms.size(); ++i)
    {
        bool i_is_not_last_observation{i < terms.size() - 1};
        if (i_is_not_last_observation)
        {
            for (size_t j = i + 1; j < terms.size(); ++j)
            {
                bool term_is_used{!is_approximately_zero(terms[i].coefficient)};
                bool other_term_is_used{!is_approximately_zero(terms[j].coefficient)};
                if (term_is_used && other_term_is_used && terms[i].equals_not_comparing_given_terms(terms[i], terms[j]))
                {
                    VectorXd values_i{terms[i].calculate(X_train)};
                    VectorXd values_j{terms[j].calculate(X_train)};
                    bool terms_are_similar{values_i == values_j};
                    if (terms_are_similar)
                    {
                        if (terms[i].get_interaction_level() > terms[j].get_interaction_level())
                        {
                            terms[j].coefficient += terms[i].coefficient;
                            terms[i].coefficient = 0;
                            break;
                        }
                        else
                        {
                            terms[i].coefficient += terms[j].coefficient;
                            terms[j].coefficient = 0;
                        }
                    }
                }
            }
        }
    }
}

void APLRRegressor::remove_unused_terms()
{
    std::vector<Term> terms_new;
    terms_new.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i)
    {
        bool term_is_used{!is_approximately_zero(terms[i].coefficient)};
        if (term_is_used)
            terms_new.push_back(terms[i]);
    }
    terms = std::move(terms_new);
}

void APLRRegressor::revert_scaling_if_using_log_link_function()
{
    if (link_function == "log")
    {
        y_train /= scaling_factor_for_log_link_function;
        intercept += std::log(1 / scaling_factor_for_log_link_function);
        for (Eigen::Index i = 0; i < intercept_steps.size(); ++i)
        {
            intercept_steps[i] += std::log(1 / scaling_factor_for_log_link_function);
        }
    }
}

void APLRRegressor::name_terms(const MatrixXd &X, const std::vector<std::string> &X_names)
{
    bool x_names_not_provided{X_names.size() == 0};
    if (x_names_not_provided)
    {
        std::vector<std::string> temp(X.cols());
        size_t X_cols{static_cast<size_t>(X.cols())};
        for (size_t i = 0; i < X_cols; ++i)
        {
            temp[i] = "X" + std::to_string(i + 1);
        }
        set_term_names(temp);
    }
    else
    {
        set_term_names(X_names);
    }
}

void APLRRegressor::set_term_names(const std::vector<std::string> &X_names)
{
    if (model_has_not_been_trained())
        throw std::runtime_error("The model must be trained with fit() before term names can be set.");

    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].name = compute_raw_base_term_name(terms[i], X_names[terms[i].base_term]);
        bool term_has_given_terms{terms[i].given_terms.size() > 0};
        if (term_has_given_terms)
        {
            terms[i].name += " * I(";
            for (size_t j = 0; j < terms[i].given_terms.size(); ++j)
            {
                terms[i].name += compute_raw_base_term_name(terms[i].given_terms[j], X_names[terms[i].given_terms[j].base_term]) + "*";
            }
            terms[i].name.pop_back();
            terms[i].name += "!=0)";
        }
        terms[i].name = "P" + std::to_string(i) + ". Interaction level: " + std::to_string(terms[i].get_interaction_level()) + ". " + terms[i].name;
    }

    term_names.resize(terms.size() + 1);
    term_coefficients.resize(terms.size() + 1);
    term_names[0] = "Intercept";
    term_coefficients[0] = intercept;
    for (size_t i = 0; i < terms.size(); ++i)
    {
        term_names[i + 1] = terms[i].name;
        term_coefficients[i + 1] = terms[i].coefficient;
    }
}

bool APLRRegressor::model_has_not_been_trained()
{
    return !std::isfinite(intercept);
}

std::string APLRRegressor::compute_raw_base_term_name(const Term &term, const std::string &X_name)
{
    std::string name{""};
    bool is_linear_effect{std::isnan(term.split_point)};
    if (is_linear_effect)
        name = X_name;
    else
    {
        double temp_split_point{term.split_point};
        std::string sign{"-"};
        if (std::isless(temp_split_point, 0))
        {
            temp_split_point = -temp_split_point;
            sign = "+";
        }
        if (term.direction_right)
            name = "max(" + X_name + sign + std::to_string(temp_split_point) + ",0)";
        else
            name = "min(" + X_name + sign + std::to_string(temp_split_point) + ",0)";
    }
    return name;
}

void APLRRegressor::calculate_feature_importance_on_validation_set()
{
    feature_importance = VectorXd::Constant(number_of_base_terms, 0);
    MatrixXd li{calculate_local_feature_importance(X_validation)};
    for (Eigen::Index i = 0; i < li.cols(); ++i) // For each column calculate mean abs values
    {
        feature_importance[i] = li.col(i).cwiseAbs().mean();
    }
}

MatrixXd APLRRegressor::calculate_local_feature_importance(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    MatrixXd output{MatrixXd::Constant(X.rows(), number_of_base_terms, 0)};

    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd contrib{terms[i].calculate_contribution_to_linear_predictor(X)};
        output.col(terms[i].base_term) += contrib;
    }

    return output;
}

void APLRRegressor::find_min_and_max_training_predictions_or_responses()
{
    VectorXd training_predictions{predict(X_train, false)};
    min_training_prediction_or_response = std::max(training_predictions.minCoeff(), y_train.minCoeff());
    max_training_prediction_or_response = std::min(training_predictions.maxCoeff(), y_train.maxCoeff());
}

void APLRRegressor::validate_that_model_can_be_used(const MatrixXd &X)
{
    if (model_has_not_been_trained())
        throw std::runtime_error("The model must be trained with fit() before predict() can be run.");
    if (X.rows() == 0)
        throw std::runtime_error("X cannot have zero rows.");
    size_t cols_provided{static_cast<size_t>(X.cols())};
    if (cols_provided != number_of_base_terms)
        throw std::runtime_error("X must have " + std::to_string(number_of_base_terms) + " columns but " + std::to_string(cols_provided) + " were provided.");
    throw_error_if_matrix_has_nan_or_infinite_elements(X, "X");
}

void APLRRegressor::cleanup_after_fit()
{
    terms.shrink_to_fit();
    X_train.resize(0, 0);
    y_train.resize(0);
    sample_weight_train.resize(0);
    X_validation.resize(0, 0);
    y_validation.resize(0);
    sample_weight_validation.resize(0);
    linear_predictor_null_model.resize(0);
    terms_eligible_current.clear();
    predictions_current.resize(0);
    predictions_current_validation.resize(0);
    neg_gradient_current.resize(0);
    linear_predictor_update.resize(0);
    linear_predictor_update_validation.resize(0);
    distributed_terms.clear();
    interactions_to_consider.clear();
    sorted_indexes_of_errors_for_interactions_to_consider.resize(0);
    linear_predictor_current.resize(0);
    linear_predictor_current_validation.resize(0);
    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].cleanup_after_fit();
    }
    predictor_indexes.clear();
    prioritized_predictors_indexes.clear();
    monotonic_constraints.clear();
    group_train.resize(0);
    group_validation.resize(0);
    unique_groups_train.clear();
    unique_groups_validation.clear();
    interaction_constraints.clear();
    interactions_eligible = 0;
    other_data_train.resize(0, 0);
    other_data_validation.resize(0, 0);
}

VectorXd APLRRegressor::predict(const MatrixXd &X, bool cap_predictions_to_minmax_in_training)
{
    validate_that_model_can_be_used(X);

    VectorXd linear_predictor{calculate_linear_predictor(X)};
    VectorXd predictions{transform_linear_predictor_to_predictions(linear_predictor, link_function, calculate_custom_transform_linear_predictor_to_predictions_function)};

    if (cap_predictions_to_minmax_in_training)
    {
        this->cap_predictions_to_minmax_in_training(predictions);
    }

    return predictions;
}

VectorXd APLRRegressor::calculate_linear_predictor(const MatrixXd &X)
{
    VectorXd linear_predictor{VectorXd::Constant(X.rows(), intercept)};
    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd contrib{terms[i].calculate_contribution_to_linear_predictor(X)};
        linear_predictor += contrib;
    }
    return linear_predictor;
}

void APLRRegressor::cap_predictions_to_minmax_in_training(VectorXd &predictions)
{
    for (Eigen::Index i = 0; i < predictions.rows(); ++i)
    {
        if (std::isgreater(predictions[i], max_training_prediction_or_response))
            predictions[i] = max_training_prediction_or_response;
        else if (std::isless(predictions[i], min_training_prediction_or_response))
            predictions[i] = min_training_prediction_or_response;
    }
}

MatrixXd APLRRegressor::calculate_local_feature_importance_for_terms(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    MatrixXd output{MatrixXd::Constant(X.rows(), terms.size(), 0)};

    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd contrib{terms[i].calculate_contribution_to_linear_predictor(X)};
        output.col(i) += contrib;
    }

    return output;
}

MatrixXd APLRRegressor::calculate_terms(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    MatrixXd output{MatrixXd::Constant(X.rows(), terms.size(), 0)};

    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd values{terms[i].calculate(X)};
        output.col(i) += values;
    }

    return output;
}

std::vector<std::string> APLRRegressor::get_term_names()
{
    return term_names;
}

VectorXd APLRRegressor::get_term_coefficients()
{
    return term_coefficients;
}

VectorXd APLRRegressor::get_term_coefficient_steps(size_t term_index)
{
    return terms[term_index].coefficient_steps;
}

VectorXd APLRRegressor::get_validation_error_steps()
{
    return validation_error_steps;
}

VectorXd APLRRegressor::get_feature_importance()
{
    return feature_importance;
}

double APLRRegressor::get_intercept()
{
    return intercept;
}

VectorXd APLRRegressor::get_intercept_steps()
{
    return intercept_steps;
}

size_t APLRRegressor::get_optimal_m()
{
    return m_optimal;
}

std::string APLRRegressor::get_validation_tuning_metric()
{
    return validation_tuning_metric;
}

std::vector<size_t> APLRRegressor::get_validation_indexes()
{
    return validation_indexes;
}