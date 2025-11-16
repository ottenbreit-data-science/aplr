#pragma once
#include <string>
#include <limits>
#include <future>
#include <random>
#include <vector>
#include <thread>
#include <unordered_map>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "functions.h"
#include "term.h"
#include "constants.h"
#include "ThreadPool.h"

using namespace Eigen;

struct ModelForCVFold
{
    double intercept;
    std::vector<Term> terms;
    MatrixXd validation_error_steps;
    double validation_error;
    size_t m_optimal;
    double sample_weight_train_sum;
    double fold_weight;
    Eigen::Index fold_index;
    double min_training_prediction_or_response;
    double max_training_prediction_or_response;
};

class APLRRegressor
{
private:
    MatrixXd X_train;
    VectorXd y_train;
    std::vector<size_t> validation_indexes;
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
    std::vector<double> predictor_learning_rates;
    std::vector<double> predictor_penalties_for_non_linearity;
    std::vector<double> predictor_penalties_for_interactions;
    std::vector<size_t> predictor_min_observations_in_split;
    VectorXi group_train;
    VectorXi group_validation;
    std::set<int> unique_groups_train;
    std::set<int> unique_groups_validation;
    std::vector<std::vector<size_t>> interaction_constraints;
    MatrixXd other_data_train;
    MatrixXd other_data_validation;
    bool model_has_changed_in_this_boosting_step;
    std::set<int> unique_prediction_groups;
    std::set<int> unique_groups_cycle_train;
    std::vector<VectorXi> group_cycle_train;
    size_t group_cycle_predictor_index;
    std::vector<ModelForCVFold> cv_fold_models;
    VectorXd intercept_steps;
    double best_validation_error_so_far;
    size_t best_m_so_far;
    bool linear_effects_only_in_this_boosting_step;
    bool non_linear_effects_allowed_in_this_boosting_step;
    bool max_terms_reached;
    bool round_robin_update_of_existing_terms;
    size_t term_to_update_in_this_boosting_step;
    size_t cores_to_use;
    std::unique_ptr<ThreadPool> thread_pool;
    bool stopped_early;
    std::vector<double> ridge_penalty_weights;
    double min_validation_error_for_current_fold;

    void validate_input_to_fit(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight, const std::vector<std::string> &X_names,
                               const MatrixXi &cv_observations, const std::vector<size_t> &prioritized_predictors_indexes,
                               const std::vector<int> &monotonic_constraints, const VectorXi &group, const std::vector<std::vector<size_t>> &interaction_constraints,
                               const MatrixXd &other_data, const std::vector<double> &predictor_learning_rates,
                               const std::vector<double> &predictor_penalties_for_non_linearity,
                               const std::vector<double> &predictor_penalties_for_interactions);
    void throw_error_if_validation_set_indexes_has_invalid_indexes(const VectorXd &y, const std::vector<size_t> &validation_set_indexes);
    void throw_error_if_prioritized_predictors_indexes_has_invalid_indexes(const MatrixXd &X, const std::vector<size_t> &prioritized_predictors_indexes);
    void throw_error_if_monotonic_constraints_has_invalid_indexes(const MatrixXd &X, const std::vector<int> &monotonic_constraints);
    void throw_error_if_predictor_penalties_or_learning_rates_have_invalid_values(const MatrixXd &X, const std::vector<double> &predictor_penaties_or_learning_rates);
    void throw_error_if_interaction_constraints_has_invalid_indexes(const MatrixXd &X, const std::vector<std::vector<size_t>> &interaction_constraints);
    MatrixXi preprocess_cv_observations(const MatrixXi &cv_observations, const VectorXd &y);
    void preprocess_prioritized_predictors_and_interaction_constraints(const MatrixXd &X, const std::vector<size_t> &prioritized_predictors_indexes,
                                                                       const std::vector<std::vector<size_t>> &interaction_constraints);
    void initialize_multithreading();
    void preprocess_penalties();
    void preprocess_penalty(double &penalty);
    void preprocess_predictor_learning_rates_and_penalties(const MatrixXd &X, const std::vector<double> &predictor_learning_rates,
                                                           const std::vector<double> &predictor_penalties_for_non_linearity,
                                                           const std::vector<double> &predictor_penalties_for_interactions);
    void preprocess_predictor_min_observations_in_split(const MatrixXd &X, const std::vector<size_t> &predictor_min_observations_in_split);
    void calculate_min_and_max_predictor_values_in_training(const MatrixXd &X);
    std::vector<double> preprocess_predictor_learning_rate_or_penalty(const MatrixXd &X, double general_value,
                                                                      const std::vector<double> &predictor_specific_values);
    void fit_model_for_cv_fold(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight,
                               const std::vector<std::string> &X_names, const VectorXi &cv_observations_in_fold,
                               const std::vector<int> &monotonic_constraints, const VectorXi &group, const MatrixXd &other_data,
                               Eigen::Index fold_index);
    void define_training_and_validation_sets(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight,
                                             const VectorXi &cv_observations_in_fold, const VectorXi &group, const MatrixXd &other_data);
    void initialize(const std::vector<int> &monotonic_constraints);
    bool check_if_base_term_has_only_one_unique_value(size_t base_term);
    void add_term_to_terms_eligible_current(Term &term);
    void setup_groups_for_group_mse_cycle();
    VectorXi create_groups_for_group_mse_sorted_by_vector(const VectorXd &vector, const std::set<int> &unique_groups_in_vector);
    VectorXd calculate_neg_gradient_current();
    VectorXd calculate_neg_gradient_current_for_group_mse(GroupData &group_residuals_and_count, const VectorXi &group,
                                                          const std::set<int> &unique_groups);
    void execute_boosting_steps(Eigen::Index fold_index);
    void execute_boosting_step(size_t boosting_step, Eigen::Index fold_index);
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
    void prepare_for_round_robin_coefficient_updates_if_max_terms_has_been_reached();
    void select_the_best_term_and_update_errors(size_t boosting_step);
    void remove_ineligibility();
    void update_terms(size_t boosting_step);
    void update_gradient_and_errors();
    void add_new_term(size_t boosting_step);
    void update_coefficient_steps(size_t boosting_step);
    double calculate_quantile_mean_response(const VectorXd &predictions, bool top_quantile);
    void calculate_and_validate_validation_error(size_t boosting_step);
    double calculate_validation_error(const VectorXd &predictions, const std::string &metric = "default");
    double calculate_group_mse_by_prediction_validation_error(const VectorXd &predictions);
    void update_term_eligibility();
    void update_a_term_coefficient_round_robin(size_t boosting_step);
    void print_summary_after_boosting_step(size_t boosting_step, Eigen::Index fold_index);
    void abort_boosting_when_no_validation_error_improvement_in_the_last_early_stopping_rounds(size_t boosting_step);
    void print_final_summary();
    void find_optimal_m_and_update_model_accordingly();
    void merge_similar_terms(const MatrixXd &X);
    void remove_unused_terms();
    void name_terms(const MatrixXd &X, const std::vector<std::string> &X_names);
    void find_min_and_max_training_predictions_or_responses();
    void write_output_to_cv_fold_models(Eigen::Index fold_index);
    void cleanup_after_fit();
    void check_term_integrity();
    void create_final_model(const MatrixXd &X, const VectorXd &sample_weight);
    void compute_fold_weights();
    void update_intercept_and_term_weights();
    void create_terms(const MatrixXd &X);
    void estimate_term_importances(const MatrixXd &X, const VectorXd &sample_weight);
    void sort_terms();
    void calculate_other_term_vectors();
    void compute_cv_error();
    void concatenate_validation_error_steps();
    void find_final_min_and_max_training_predictions_or_responses();
    void compute_max_optimal_m();
    void correct_term_names_coefficients_and_affiliations();
    void additional_cleanup_after_creating_final_model();
    void validate_that_model_can_be_used(const MatrixXd &X);
    void throw_error_if_loss_function_does_not_exist();
    void throw_error_if_link_function_does_not_exist();
    VectorXd calculate_linear_predictor(const MatrixXd &X);
    void update_linear_predictor_and_predictions();
    void throw_error_if_response_contains_invalid_values(const VectorXd &y);
    void throw_error_if_sample_weight_contains_invalid_values(const VectorXd &y, const VectorXd &sample_weight);
    void throw_error_if_response_is_not_between_0_and_1(const VectorXd &y, const std::string &error_message);
    void throw_error_if_vector_contains_negative_values(const VectorXd &y, const std::string &error_message);
    void throw_error_if_vector_contains_non_positive_values(const VectorXd &y, const std::string &error_message);
    void throw_error_if_dispersion_parameter_is_invalid();
    VectorXd differentiate_predictions_wrt_linear_predictor();
    void correct_mean_bias();
    void scale_response_if_using_log_link_function();
    void revert_scaling_if_using_log_link_function();
    void cap_predictions_to_minmax_in_training(VectorXd &predictions);
    std::string compute_raw_base_term_name(const Term &term, const std::string &X_name);
    void throw_error_if_m_is_invalid();
    bool model_has_not_been_trained();
    void throw_error_if_quantile_is_invalid();
    void throw_error_if_validation_tuning_metric_is_invalid();
    std::vector<size_t> compute_relevant_term_indexes(const std::string &unique_term_affiliation);
    std::vector<double> compute_split_points(size_t predictor_index, const std::vector<size_t> &relevant_term_indexes);
    void validate_fold_index(size_t fold_index);
    void throw_cv_data_not_available_error(const std::string &data_name);
    VectorXd compute_contribution_to_linear_predictor_from_specific_terms(const MatrixXd &X, const std::vector<size_t> &term_indexes,
                                                                          const std::vector<size_t> &base_predictors_used);
    void validate_sample_weight(const MatrixXd &X, const VectorXd &sample_weight);
    void set_term_coefficients();

public:
    double intercept;
    std::vector<Term> terms;
    size_t m;
    size_t m_optimal;
    double v;
    std::string loss_function;
    std::string link_function;
    size_t cv_folds;
    size_t n_jobs;
    uint_fast32_t random_state;
    size_t bins;
    size_t verbosity;
    std::vector<std::string> term_names;
    std::vector<std::string> term_affiliations;
    VectorXd term_coefficients;
    size_t max_interaction_level;
    size_t max_interactions;
    size_t interactions_eligible;
    MatrixXd validation_error_steps;
    size_t min_observations_in_split;
    size_t ineligible_boosting_steps_added;
    size_t max_eligible_terms;
    size_t number_of_base_terms;
    size_t number_of_unique_term_affiliations;
    std::vector<std::string> unique_term_affiliations;
    std::map<std::string, size_t> unique_term_affiliation_map;
    std::vector<std::vector<size_t>> base_predictors_in_each_unique_term_affiliation;
    VectorXd feature_importance;
    VectorXd term_importance;
    double dispersion_parameter;
    double min_training_prediction_or_response;
    double max_training_prediction_or_response;
    std::string validation_tuning_metric;
    double quantile;
    std::function<double(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group, const MatrixXd &other_data)> calculate_custom_validation_error_function;
    std::function<double(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group, const MatrixXd &other_data)> calculate_custom_loss_function;
    std::function<VectorXd(const VectorXd &y, const VectorXd &predictions, const VectorXi &group, const MatrixXd &other_data)> calculate_custom_negative_gradient_function;
    std::function<VectorXd(const VectorXd &linear_predictor)> calculate_custom_transform_linear_predictor_to_predictions_function;
    std::function<VectorXd(const VectorXd &linear_predictor)> calculate_custom_differentiate_predictions_wrt_linear_predictor_function;
    size_t boosting_steps_before_interactions_are_allowed;
    bool monotonic_constraints_ignore_interactions;
    size_t group_mse_by_prediction_bins;
    size_t group_mse_cycle_min_obs_in_bin;
    double cv_error;
    VectorXi term_main_predictor_indexes;
    VectorXi term_interaction_levels;
    size_t early_stopping_rounds;
    size_t num_first_steps_with_linear_effects_only;
    double penalty_for_non_linearity;
    double penalty_for_interactions;
    size_t max_terms;
    VectorXd min_predictor_values_in_training;
    VectorXd max_predictor_values_in_training;
    double ridge_penalty;
    bool mean_bias_correction;
    bool faster_convergence;

    std::vector<VectorXd> cv_validation_predictions_all_folds;
    std::vector<VectorXd> cv_y_all_folds;
    std::vector<VectorXd> cv_sample_weight_all_folds;
    std::vector<VectorXi> cv_validation_indexes_all_folds;

    APLRRegressor(size_t m = 3000, double v = 0.5, uint_fast32_t random_state = std::numeric_limits<uint_fast32_t>::lowest(), std::string loss_function = "mse",
                  std::string link_function = "identity", size_t n_jobs = 0, size_t cv_folds = 5,
                  size_t bins = 300, size_t verbosity = 0, size_t max_interaction_level = 1, size_t max_interactions = 100000,
                  size_t min_observations_in_split = 4, size_t ineligible_boosting_steps_added = 15, size_t max_eligible_terms = 7, double dispersion_parameter = 1.5,
                  std::string validation_tuning_metric = "default", double quantile = 0.5,
                  const std::function<double(VectorXd, VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_validation_error_function = {},
                  const std::function<double(VectorXd, VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_loss_function = {},
                  const std::function<VectorXd(VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_negative_gradient_function = {},
                  const std::function<VectorXd(VectorXd)> &calculate_custom_transform_linear_predictor_to_predictions_function = {},
                  const std::function<VectorXd(VectorXd)> &calculate_custom_differentiate_predictions_wrt_linear_predictor_function = {},
                  size_t boosting_steps_before_interactions_are_allowed = 0, bool monotonic_constraints_ignore_interactions = false,
                  size_t group_mse_by_prediction_bins = 10, size_t group_mse_cycle_min_obs_in_bin = 30, size_t early_stopping_rounds = 200,
                  size_t num_first_steps_with_linear_effects_only = 0, double penalty_for_non_linearity = 0.0, double penalty_for_interactions = 0.0, size_t max_terms = 0,
                  double ridge_penalty = 0.0001, bool mean_bias_correction = false, bool faster_convergence = false);
    APLRRegressor(const APLRRegressor &other);
    APLRRegressor &operator=(const APLRRegressor &other);
    ~APLRRegressor();
    void fit(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight = VectorXd(0), const std::vector<std::string> &X_names = {},
             const MatrixXi &cv_observations = MatrixXi(0, 0), const std::vector<size_t> &prioritized_predictors_indexes = {},
             const std::vector<int> &monotonic_constraints = {}, const VectorXi &group = VectorXi(0), const std::vector<std::vector<size_t>> &interaction_constraints = {},
             const MatrixXd &other_data = MatrixXd(0, 0), const std::vector<double> &predictor_learning_rates = {},
             const std::vector<double> &predictor_penalties_for_non_linearity = {},
             const std::vector<double> &predictor_penalties_for_interactions = {},
             const std::vector<size_t> &predictor_min_observations_in_split = {});
    VectorXd predict(const MatrixXd &X, bool cap_predictions_to_minmax_in_training = true);
    void set_term_names(const std::vector<std::string> &X_names);
    void set_term_affiliations(const std::vector<std::string> &X_names);
    VectorXd calculate_feature_importance(const MatrixXd &X, const VectorXd &sample_weight = VectorXd(0));
    VectorXd calculate_term_importance(const MatrixXd &X, const VectorXd &sample_weight = VectorXd(0));
    MatrixXd calculate_local_feature_contribution(const MatrixXd &X);
    MatrixXd calculate_local_term_contribution(const MatrixXd &X);
    VectorXd calculate_local_contribution_from_selected_terms(const MatrixXd &X, const std::vector<size_t> &predictor_indexes);
    MatrixXd calculate_terms(const MatrixXd &X);
    std::vector<std::string> get_term_names();
    std::vector<std::string> get_term_affiliations();
    std::vector<std::string> get_unique_term_affiliations();
    std::vector<std::vector<size_t>> get_base_predictors_in_each_unique_term_affiliation();
    VectorXd get_term_coefficients();
    MatrixXd get_validation_error_steps();
    VectorXd get_feature_importance();
    VectorXd get_term_importance();
    VectorXi get_term_main_predictor_indexes();
    VectorXi get_term_interaction_levels();
    double get_intercept();
    size_t get_optimal_m();
    std::string get_validation_tuning_metric();
    std::map<double, double> get_main_effect_shape(size_t predictor_index);
    MatrixXd get_unique_term_affiliation_shape(const std::string &unique_term_affiliation, size_t max_rows_before_sampling = 500000, size_t additional_points = 250);
    MatrixXd generate_predictor_values_and_contribution(const std::vector<size_t> &relevant_term_indexes,
                                                        size_t unique_term_affiliation_index);
    double get_cv_error();
    size_t get_num_cv_folds();
    void set_intercept(double value);
    void remove_provided_custom_functions();

    VectorXd get_cv_validation_predictions(size_t fold_index);
    VectorXd get_cv_y(size_t fold_index);
    VectorXd get_cv_sample_weight(size_t fold_index);
    VectorXi get_cv_validation_indexes(size_t fold_index);
    void clear_cv_results();

    friend class APLRClassifier;
};

APLRRegressor::APLRRegressor(size_t m, double v, uint_fast32_t random_state, std::string loss_function, std::string link_function, size_t n_jobs,
                             size_t cv_folds, size_t bins, size_t verbosity, size_t max_interaction_level,
                             size_t max_interactions, size_t min_observations_in_split, size_t ineligible_boosting_steps_added, size_t max_eligible_terms, double dispersion_parameter,
                             std::string validation_tuning_metric, double quantile,
                             const std::function<double(VectorXd, VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_validation_error_function,
                             const std::function<double(VectorXd, VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_loss_function,
                             const std::function<VectorXd(VectorXd, VectorXd, VectorXi, MatrixXd)> &calculate_custom_negative_gradient_function,
                             const std::function<VectorXd(VectorXd)> &calculate_custom_transform_linear_predictor_to_predictions_function,
                             const std::function<VectorXd(VectorXd)> &calculate_custom_differentiate_predictions_wrt_linear_predictor_function,
                             size_t boosting_steps_before_interactions_are_allowed, bool monotonic_constraints_ignore_interactions,
                             size_t group_mse_by_prediction_bins, size_t group_mse_cycle_min_obs_in_bin, size_t early_stopping_rounds,
                             size_t num_first_steps_with_linear_effects_only, double penalty_for_non_linearity, double penalty_for_interactions, size_t max_terms, double ridge_penalty,
                             bool mean_bias_correction, bool faster_convergence)
    : intercept{NAN_DOUBLE}, m{m}, v{v},
      loss_function{loss_function}, link_function{link_function}, cv_folds{cv_folds}, n_jobs{n_jobs}, random_state{random_state},
      bins{bins}, verbosity{verbosity}, max_interaction_level{max_interaction_level},
      max_interactions{max_interactions}, interactions_eligible{0}, validation_error_steps{MatrixXd(0, 0)},
      min_observations_in_split{min_observations_in_split}, ineligible_boosting_steps_added{ineligible_boosting_steps_added},
      max_eligible_terms{max_eligible_terms}, number_of_base_terms{0}, number_of_unique_term_affiliations{0},
      dispersion_parameter{dispersion_parameter}, min_training_prediction_or_response{NAN_DOUBLE},
      max_training_prediction_or_response{NAN_DOUBLE}, validation_tuning_metric{validation_tuning_metric},
      quantile{quantile}, calculate_custom_validation_error_function{calculate_custom_validation_error_function},
      calculate_custom_loss_function{calculate_custom_loss_function}, calculate_custom_negative_gradient_function{calculate_custom_negative_gradient_function},
      calculate_custom_transform_linear_predictor_to_predictions_function{calculate_custom_transform_linear_predictor_to_predictions_function},
      calculate_custom_differentiate_predictions_wrt_linear_predictor_function{calculate_custom_differentiate_predictions_wrt_linear_predictor_function},
      boosting_steps_before_interactions_are_allowed{boosting_steps_before_interactions_are_allowed},
      monotonic_constraints_ignore_interactions{monotonic_constraints_ignore_interactions}, group_mse_by_prediction_bins{group_mse_by_prediction_bins},
      group_mse_cycle_min_obs_in_bin{group_mse_cycle_min_obs_in_bin}, cv_error{NAN_DOUBLE}, early_stopping_rounds{early_stopping_rounds},
      num_first_steps_with_linear_effects_only{num_first_steps_with_linear_effects_only}, penalty_for_non_linearity{penalty_for_non_linearity},
      penalty_for_interactions{penalty_for_interactions}, max_terms{max_terms}, ridge_penalty{ridge_penalty}, mean_bias_correction{mean_bias_correction},
      faster_convergence{faster_convergence}
{
}

APLRRegressor::APLRRegressor(const APLRRegressor &other)
    : intercept{other.intercept}, terms{other.terms}, m{other.m}, v{other.v},
      loss_function{other.loss_function}, link_function{other.link_function}, cv_folds{other.cv_folds},
      n_jobs{other.n_jobs}, random_state{other.random_state}, bins{other.bins},
      verbosity{other.verbosity}, term_names{other.term_names}, term_affiliations{other.term_affiliations}, term_coefficients{other.term_coefficients},
      max_interaction_level{other.max_interaction_level}, max_interactions{other.max_interactions},
      interactions_eligible{other.interactions_eligible}, validation_error_steps{other.validation_error_steps},
      min_observations_in_split{other.min_observations_in_split}, ineligible_boosting_steps_added{other.ineligible_boosting_steps_added},
      max_eligible_terms{other.max_eligible_terms}, number_of_base_terms{other.number_of_base_terms},
      number_of_unique_term_affiliations{other.number_of_unique_term_affiliations},
      feature_importance{other.feature_importance}, term_importance{other.term_importance}, dispersion_parameter{other.dispersion_parameter},
      min_training_prediction_or_response{other.min_training_prediction_or_response},
      max_training_prediction_or_response{other.max_training_prediction_or_response}, validation_tuning_metric{other.validation_tuning_metric},
      quantile{other.quantile}, m_optimal{other.m_optimal},
      calculate_custom_validation_error_function{other.calculate_custom_validation_error_function},
      calculate_custom_loss_function{other.calculate_custom_loss_function}, calculate_custom_negative_gradient_function{other.calculate_custom_negative_gradient_function},
      calculate_custom_transform_linear_predictor_to_predictions_function{other.calculate_custom_transform_linear_predictor_to_predictions_function},
      calculate_custom_differentiate_predictions_wrt_linear_predictor_function{other.calculate_custom_differentiate_predictions_wrt_linear_predictor_function},
      boosting_steps_before_interactions_are_allowed{other.boosting_steps_before_interactions_are_allowed},
      monotonic_constraints_ignore_interactions{other.monotonic_constraints_ignore_interactions}, group_mse_by_prediction_bins{other.group_mse_by_prediction_bins},
      group_mse_cycle_min_obs_in_bin{other.group_mse_cycle_min_obs_in_bin}, cv_error{other.cv_error},
      term_main_predictor_indexes{other.term_main_predictor_indexes}, term_interaction_levels{other.term_interaction_levels},
      early_stopping_rounds{other.early_stopping_rounds},
      num_first_steps_with_linear_effects_only{other.num_first_steps_with_linear_effects_only},
      penalty_for_non_linearity{other.penalty_for_non_linearity}, penalty_for_interactions{other.penalty_for_interactions},
      max_terms{other.max_terms}, min_predictor_values_in_training{other.min_predictor_values_in_training},
      max_predictor_values_in_training{other.max_predictor_values_in_training}, unique_term_affiliations{other.unique_term_affiliations},
      unique_term_affiliation_map{other.unique_term_affiliation_map},
      base_predictors_in_each_unique_term_affiliation{other.base_predictors_in_each_unique_term_affiliation}, ridge_penalty{other.ridge_penalty},
      mean_bias_correction{other.mean_bias_correction}, faster_convergence{other.faster_convergence},
      cv_validation_predictions_all_folds{other.cv_validation_predictions_all_folds},
      cv_y_all_folds{other.cv_y_all_folds}, cv_sample_weight_all_folds{other.cv_sample_weight_all_folds},
      cv_validation_indexes_all_folds{other.cv_validation_indexes_all_folds}
{
}

APLRRegressor &APLRRegressor::operator=(const APLRRegressor &other)
{
    if (this == &other)
    {
        return *this;
    }

    intercept = other.intercept;
    terms = other.terms;
    m = other.m;
    v = other.v;
    loss_function = other.loss_function;
    link_function = other.link_function;
    cv_folds = other.cv_folds;
    n_jobs = other.n_jobs;
    random_state = other.random_state;
    bins = other.bins;
    verbosity = other.verbosity;
    term_names = other.term_names;
    term_affiliations = other.term_affiliations;
    term_coefficients = other.term_coefficients;
    max_interaction_level = other.max_interaction_level;
    max_interactions = other.max_interactions;
    interactions_eligible = other.interactions_eligible;
    validation_error_steps = other.validation_error_steps;
    min_observations_in_split = other.min_observations_in_split;
    ineligible_boosting_steps_added = other.ineligible_boosting_steps_added;
    max_eligible_terms = other.max_eligible_terms;
    number_of_base_terms = other.number_of_base_terms;
    number_of_unique_term_affiliations = other.number_of_unique_term_affiliations;
    feature_importance = other.feature_importance;
    term_importance = other.term_importance;
    dispersion_parameter = other.dispersion_parameter;
    min_training_prediction_or_response = other.min_training_prediction_or_response;
    max_training_prediction_or_response = other.max_training_prediction_or_response;
    validation_tuning_metric = other.validation_tuning_metric;
    quantile = other.quantile;
    m_optimal = other.m_optimal;
    calculate_custom_validation_error_function = other.calculate_custom_validation_error_function;
    calculate_custom_loss_function = other.calculate_custom_loss_function;
    calculate_custom_negative_gradient_function = other.calculate_custom_negative_gradient_function;
    calculate_custom_transform_linear_predictor_to_predictions_function = other.calculate_custom_transform_linear_predictor_to_predictions_function;
    calculate_custom_differentiate_predictions_wrt_linear_predictor_function = other.calculate_custom_differentiate_predictions_wrt_linear_predictor_function;
    boosting_steps_before_interactions_are_allowed = other.boosting_steps_before_interactions_are_allowed;
    monotonic_constraints_ignore_interactions = other.monotonic_constraints_ignore_interactions;
    group_mse_by_prediction_bins = other.group_mse_by_prediction_bins;
    group_mse_cycle_min_obs_in_bin = other.group_mse_cycle_min_obs_in_bin;
    cv_error = other.cv_error;
    term_main_predictor_indexes = other.term_main_predictor_indexes;
    term_interaction_levels = other.term_interaction_levels;
    early_stopping_rounds = other.early_stopping_rounds;
    num_first_steps_with_linear_effects_only = other.num_first_steps_with_linear_effects_only;
    penalty_for_non_linearity = other.penalty_for_non_linearity;
    penalty_for_interactions = other.penalty_for_interactions;
    max_terms = other.max_terms;
    min_predictor_values_in_training = other.min_predictor_values_in_training;
    max_predictor_values_in_training = other.max_predictor_values_in_training;
    unique_term_affiliations = other.unique_term_affiliations;
    unique_term_affiliation_map = other.unique_term_affiliation_map;
    base_predictors_in_each_unique_term_affiliation = other.base_predictors_in_each_unique_term_affiliation;
    ridge_penalty = other.ridge_penalty;
    mean_bias_correction = other.mean_bias_correction;
    faster_convergence = other.faster_convergence;
    cv_validation_predictions_all_folds = other.cv_validation_predictions_all_folds;
    cv_y_all_folds = other.cv_y_all_folds;
    cv_sample_weight_all_folds = other.cv_sample_weight_all_folds;
    cv_validation_indexes_all_folds = other.cv_validation_indexes_all_folds;

    thread_pool.reset();

    return *this;
}

APLRRegressor::~APLRRegressor()
{
}

void APLRRegressor::fit(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight, const std::vector<std::string> &X_names,
                        const MatrixXi &cv_observations, const std::vector<size_t> &prioritized_predictors_indexes,
                        const std::vector<int> &monotonic_constraints, const VectorXi &group, const std::vector<std::vector<size_t>> &interaction_constraints,
                        const MatrixXd &other_data, const std::vector<double> &predictor_learning_rates,
                        const std::vector<double> &predictor_penalties_for_non_linearity,
                        const std::vector<double> &predictor_penalties_for_interactions,
                        const std::vector<size_t> &predictor_min_observations_in_split)
{
    throw_error_if_loss_function_does_not_exist();
    throw_error_if_link_function_does_not_exist();
    throw_error_if_dispersion_parameter_is_invalid();
    throw_error_if_quantile_is_invalid();
    throw_error_if_m_is_invalid();
    throw_error_if_validation_tuning_metric_is_invalid();
    validate_input_to_fit(X, y, sample_weight, X_names, cv_observations, prioritized_predictors_indexes, monotonic_constraints, group,
                          interaction_constraints, other_data, predictor_learning_rates, predictor_penalties_for_non_linearity,
                          predictor_penalties_for_interactions);

    VectorXd sample_weight_used{sample_weight};
    if (sample_weight.size() == 0)
    {
        sample_weight_used = VectorXd::Constant(y.rows(), 1.0);
    }
    sample_weight_used /= sample_weight_used.mean();

    MatrixXi cv_observations_used{preprocess_cv_observations(cv_observations, y)};
    preprocess_prioritized_predictors_and_interaction_constraints(X, prioritized_predictors_indexes, interaction_constraints);
    initialize_multithreading();
    preprocess_penalties();
    preprocess_predictor_learning_rates_and_penalties(X, predictor_learning_rates, predictor_penalties_for_non_linearity,
                                                      predictor_penalties_for_interactions);
    preprocess_predictor_min_observations_in_split(X, predictor_min_observations_in_split);
    calculate_min_and_max_predictor_values_in_training(X);
    cv_fold_models.resize(cv_observations_used.cols());
    cv_validation_predictions_all_folds.resize(cv_observations_used.cols());
    cv_y_all_folds.resize(cv_observations_used.cols());
    cv_sample_weight_all_folds.resize(cv_observations_used.cols());
    cv_validation_indexes_all_folds.resize(cv_observations_used.cols());

    for (Eigen::Index i = 0; i < cv_observations_used.cols(); ++i)
    {
        fit_model_for_cv_fold(X, y, sample_weight_used, X_names, cv_observations_used.col(i), monotonic_constraints, group, other_data, i);
    }
    create_final_model(X, sample_weight_used);
}

void APLRRegressor::preprocess_prioritized_predictors_and_interaction_constraints(
    const MatrixXd &X,
    const std::vector<size_t> &prioritized_predictors_indexes,
    const std::vector<std::vector<size_t>> &interaction_constraints)
{
    predictor_indexes.resize(X.cols());
    for (size_t i = 0; i < X.cols(); ++i)
    {
        predictor_indexes[i] = i;
    }
    this->prioritized_predictors_indexes = prioritized_predictors_indexes;

    this->interaction_constraints = interaction_constraints;
    for (auto &legal_interaction_combination : this->interaction_constraints)
    {
        legal_interaction_combination = remove_duplicate_elements_from_vector(legal_interaction_combination);
    }
}

void APLRRegressor::initialize_multithreading()
{
    size_t available_cores{static_cast<size_t>(std::thread::hardware_concurrency())};
    if (n_jobs == 0)
        cores_to_use = available_cores;
    else
        cores_to_use = std::min(n_jobs, available_cores);
    if (cores_to_use > 1)
        thread_pool = std::make_unique<ThreadPool>(cores_to_use);
}

void APLRRegressor::preprocess_penalties()
{
    preprocess_penalty(penalty_for_non_linearity);
    preprocess_penalty(penalty_for_interactions);
}

void APLRRegressor::preprocess_penalty(double &penalty)
{
    if (std::isgreater(penalty, 1.0))
        penalty = 1.0;
    else if (std::isless(penalty, 0.0))
        penalty = 0.0;
}

void APLRRegressor::preprocess_predictor_learning_rates_and_penalties(const MatrixXd &X,
                                                                      const std::vector<double> &predictor_learning_rates,
                                                                      const std::vector<double> &predictor_penalties_for_non_linearity,
                                                                      const std::vector<double> &predictor_penalties_for_interactions)
{
    this->predictor_learning_rates = preprocess_predictor_learning_rate_or_penalty(X, v, predictor_learning_rates);
    this->predictor_penalties_for_non_linearity = preprocess_predictor_learning_rate_or_penalty(X, penalty_for_non_linearity,
                                                                                                predictor_penalties_for_non_linearity);
    this->predictor_penalties_for_interactions = preprocess_predictor_learning_rate_or_penalty(X, penalty_for_interactions,
                                                                                               predictor_penalties_for_interactions);
}

std::vector<double> APLRRegressor::preprocess_predictor_learning_rate_or_penalty(const MatrixXd &X, double general_value,
                                                                                 const std::vector<double> &predictor_specific_values)
{
    std::vector<double> output(X.cols());
    bool predictor_specific_values_are_provided{predictor_specific_values.size() > 0};
    if (predictor_specific_values_are_provided)
    {
        output = predictor_specific_values;
    }
    else
    {
        for (size_t i = 0; i < output.size(); ++i)
        {
            output[i] = general_value;
        }
    }
    return output;
}

void APLRRegressor::preprocess_predictor_min_observations_in_split(
    const MatrixXd &X,
    const std::vector<size_t> &predictor_min_observations_in_split)
{
    if (predictor_min_observations_in_split.empty())
    {
        this->predictor_min_observations_in_split = std::vector<size_t>(X.cols(), min_observations_in_split);
    }
    else if (predictor_min_observations_in_split.size() != X.cols())
    {
        throw std::runtime_error("The size of predictor_min_observations_in_split does not match the number of columns in X.");
    }
    else
    {
        this->predictor_min_observations_in_split = predictor_min_observations_in_split;
    }
}

void APLRRegressor::calculate_min_and_max_predictor_values_in_training(const MatrixXd &X)
{
    min_predictor_values_in_training = VectorXd(X.cols());
    max_predictor_values_in_training = VectorXd(X.cols());
    for (Eigen::Index i = 0; i < X.cols(); ++i)
    {
        min_predictor_values_in_training[i] = X.col(i).minCoeff();
        max_predictor_values_in_training[i] = X.col(i).maxCoeff();
    }
}

void APLRRegressor::fit_model_for_cv_fold(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight,
                                          const std::vector<std::string> &X_names, const VectorXi &cv_observations_in_fold,
                                          const std::vector<int> &monotonic_constraints, const VectorXi &group, const MatrixXd &other_data,
                                          Eigen::Index fold_index)
{
    define_training_and_validation_sets(X, y, sample_weight, cv_observations_in_fold, group, other_data);
    scale_response_if_using_log_link_function();
    initialize(monotonic_constraints);
    execute_boosting_steps(fold_index);
    print_final_summary();
    find_optimal_m_and_update_model_accordingly();
    merge_similar_terms(X_train);
    remove_unused_terms();
    if (mean_bias_correction)
        correct_mean_bias();
    revert_scaling_if_using_log_link_function();
    set_term_coefficients();
    name_terms(X, X_names);
    VectorXd predictions_validation{predict(X_validation, false)};
    cv_validation_predictions_all_folds[fold_index] = predictions_validation;
    cv_y_all_folds[fold_index] = y_validation;
    cv_sample_weight_all_folds[fold_index] = sample_weight_validation;
    cv_validation_indexes_all_folds[fold_index] = Eigen::Map<const Vector<size_t, -1>>(validation_indexes.data(), validation_indexes.size()).cast<int>();
    min_validation_error_for_current_fold = calculate_validation_error(predictions_validation, validation_tuning_metric);
    find_min_and_max_training_predictions_or_responses();
    write_output_to_cv_fold_models(fold_index);
    cleanup_after_fit();
    // check_term_integrity();
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
    else if (loss_function == "group_mse_cycle")
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
    else if (loss_function == "huber")
        loss_function_exists = true;
    else if (loss_function == "exponential_power")
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
    else if (loss_function == "negative_binomial" || loss_function == "cauchy" || loss_function == "weibull" || loss_function == "huber" || loss_function == "exponential_power")
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

void APLRRegressor::throw_error_if_quantile_is_invalid()
{
    if (loss_function == "quantile" || validation_tuning_metric == "neg_top_quantile_mean_response" || validation_tuning_metric == "bottom_quantile_mean_response")
    {
        if (quantile < 0.0 || quantile > 1.0)
        {
            throw std::runtime_error("Quantile must be between 0.0 and 1.0.");
        }
    }
}

void APLRRegressor::throw_error_if_validation_tuning_metric_is_invalid()
{
    bool metric_exists{false};
    if (validation_tuning_metric == "default")
        metric_exists = true;
    else if (validation_tuning_metric == "mse")
        metric_exists = true;
    else if (validation_tuning_metric == "mae")
        metric_exists = true;
    else if (validation_tuning_metric == "huber")
        metric_exists = true;
    else if (validation_tuning_metric == "negative_gini")
        metric_exists = true;
    else if (validation_tuning_metric == "group_mse")
        metric_exists = true;
    else if (validation_tuning_metric == "group_mse_by_prediction")
        metric_exists = true;
    else if (validation_tuning_metric == "neg_top_quantile_mean_response")
        metric_exists = true;
    else if (validation_tuning_metric == "bottom_quantile_mean_response")
        metric_exists = true;
    else if (validation_tuning_metric == "custom_function")
        metric_exists = true;

    if (!metric_exists)
        throw std::runtime_error("validation_tuning_metric " + validation_tuning_metric + " is not available in APLR.");
}

void APLRRegressor::validate_input_to_fit(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight,
                                          const std::vector<std::string> &X_names, const MatrixXi &cv_observations,
                                          const std::vector<size_t> &prioritized_predictors_indexes, const std::vector<int> &monotonic_constraints, const VectorXi &group,
                                          const std::vector<std::vector<size_t>> &interaction_constraints, const MatrixXd &other_data,
                                          const std::vector<double> &predictor_learning_rates,
                                          const std::vector<double> &predictor_penalties_for_non_linearity,
                                          const std::vector<double> &predictor_penalties_for_interactions)
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
    throw_error_if_prioritized_predictors_indexes_has_invalid_indexes(X, prioritized_predictors_indexes);
    throw_error_if_monotonic_constraints_has_invalid_indexes(X, monotonic_constraints);
    throw_error_if_predictor_penalties_or_learning_rates_have_invalid_values(X, predictor_learning_rates);
    throw_error_if_predictor_penalties_or_learning_rates_have_invalid_values(X, predictor_penalties_for_non_linearity);
    throw_error_if_predictor_penalties_or_learning_rates_have_invalid_values(X, predictor_penalties_for_interactions);
    throw_error_if_interaction_constraints_has_invalid_indexes(X, interaction_constraints);
    throw_error_if_response_contains_invalid_values(y);
    throw_error_if_sample_weight_contains_invalid_values(y, sample_weight);
    bool cv_observations_is_provided{cv_observations.size() > 0};
    if (cv_observations_is_provided)
    {
        bool incorrect_size{cv_observations.rows() != y.rows()};
        if (incorrect_size)
            throw std::runtime_error("If cv_observations is provided then it must have as many rows as X.");
        for (Eigen::Index i = 0; i < cv_observations.cols(); ++i)
        {
            Eigen::Index rows_with_ones{(cv_observations.col(i).array() == 1).count()};
            Eigen::Index rows_with_minus_ones{(cv_observations.col(i).array() == -1).count()};
            if (rows_with_ones < MIN_OBSERATIONS_IN_A_CV_FOLD || rows_with_minus_ones < MIN_OBSERATIONS_IN_A_CV_FOLD)
                throw std::runtime_error("Each column in cv_observations must contain at least " + std::to_string(MIN_OBSERATIONS_IN_A_CV_FOLD) + " observations for each of the values 1 and -1.");
        }
    }
    bool group_is_of_incorrect_size{(loss_function == "group_mse" || validation_tuning_metric == "group_mse") && group.rows() != y.rows()};
    if (group_is_of_incorrect_size)
        throw std::runtime_error("When loss_function or validation_tuning_metric is group_mse then y and group must have the same number of rows.");
    bool other_data_is_provided{other_data.size() > 0};
    if (other_data_is_provided)
    {
        bool other_data_is_of_incorrect_size{other_data.rows() != y.rows()};
        if (other_data_is_of_incorrect_size)
            throw std::runtime_error("other_data and y must have the same number of rows.");
    }
    bool group_mse_cycle_is_used{loss_function == "group_mse_cycle" || validation_tuning_metric == "group_mse_cycle"};
    if (group_mse_cycle_is_used)
    {
        bool group_mse_by_prediction_bins_is_too_low{group_mse_by_prediction_bins < 2};
        if (group_mse_by_prediction_bins_is_too_low)
            group_mse_by_prediction_bins = 2;
        bool group_mse_cycle_min_obs_in_bin_is_too_low{group_mse_cycle_min_obs_in_bin < 1};
        if (group_mse_cycle_min_obs_in_bin_is_too_low)
            group_mse_cycle_min_obs_in_bin = 1;
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

void APLRRegressor::throw_error_if_predictor_penalties_or_learning_rates_have_invalid_values(const MatrixXd &X,
                                                                                             const std::vector<double> &predictor_penaties_or_learning_rates)
{
    bool is_provided{predictor_penaties_or_learning_rates.size() > 0};
    if (is_provided)
    {
        bool dimension_error{predictor_penaties_or_learning_rates.size() != X.cols()};
        if (dimension_error)
            throw std::runtime_error("predictor specific penalties or learning rates must either be empty or a vector with a float value for each column in X.");
        for (auto &value : predictor_penaties_or_learning_rates)
        {
            if (std::isless(value, 0.0) || std::isgreater(value, 1.0))
                throw std::runtime_error("predictor specific penalties or learning rates must not be less than zero or greater than one.");
        }
    }
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
    else if (loss_function == "gamma" || (loss_function == "tweedie" && std::isgreater(dispersion_parameter, 2.0)))
    {
        std::string error_message;
        if (loss_function == "tweedie")
            error_message = "Response values for the " + loss_function + " loss_function when dispersion_parameter>2 must be greater than zero.";
        else
            error_message = "Response values for the " + loss_function + " loss_function must be greater than zero.";
        throw_error_if_vector_contains_non_positive_values(y, error_message);
    }
    else if (link_function == "log" || loss_function == "poisson" || loss_function == "negative_binomial" || loss_function == "weibull" || (loss_function == "tweedie" && std::isless(dispersion_parameter, 2.0) && std::isgreater(dispersion_parameter, 1.0)))
    {
        std::string error_message{"Response values for the log link function or poisson loss_function or negative binomial loss function or weibull loss function or tweedie loss_function when dispersion_parameter<2 cannot be less than zero."};
        throw_error_if_vector_contains_negative_values(y, error_message);
    }
    else if (validation_tuning_metric == "negative_gini")
    {
        std::string error_message{"Response values cannot be negative when using the negative_gini validation_tuning_metric."};
        throw_error_if_vector_contains_negative_values(y, error_message);
        bool sum_is_zero{is_approximately_zero(y.sum())};
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

void APLRRegressor::throw_error_if_vector_contains_non_positive_values(const VectorXd &y, const std::string &error_message)
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
    }
}

MatrixXi APLRRegressor::preprocess_cv_observations(const MatrixXi &cv_observations, const VectorXd &y)
{
    bool cv_observations_not_provided{cv_observations.size() == 0};
    MatrixXi output{MatrixXi(0, 0)};
    if (cv_observations_not_provided)
    {
        bool invalid_cv_folds{cv_folds < 2};
        if (invalid_cv_folds)
            throw std::runtime_error("cv_folds must be at least 2.");
        output = MatrixXi::Constant(y.rows(), cv_folds, 1);
        VectorXd cv_fold{VectorXd(y.rows())};
        std::mt19937 mersenne{random_state};
        std::uniform_int_distribution<int> distribution(0, cv_folds - 1);
        for (Eigen::Index i = 0; i < y.size(); ++i)
        {
            int roll{distribution(mersenne)};
            cv_fold[i] = roll;
        }
        for (Eigen::Index i = 0; i < cv_fold.size(); ++i)
        {

            output.col(cv_fold[i])[i] = -1;
        }
        for (Eigen::Index i = 0; i < output.cols(); ++i)
        {
            Eigen::Index rows_with_ones{(output.col(i).array() == 1).count()};
            Eigen::Index rows_with_minus_ones{(output.col(i).array() == -1).count()};
            if (rows_with_ones < MIN_OBSERATIONS_IN_A_CV_FOLD || rows_with_minus_ones < MIN_OBSERATIONS_IN_A_CV_FOLD)
                throw std::runtime_error("Did not generate enough observations in a fold. Please try again with a different random_state and/or change cv_folds.");
        }
    }
    else
    {
        output = cv_observations;
    }
    return output;
}

void APLRRegressor::define_training_and_validation_sets(const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight,
                                                        const VectorXi &cv_observations_in_fold, const VectorXi &group,
                                                        const MatrixXd &other_data)
{
    size_t y_size{static_cast<size_t>(y.size())};
    std::vector<size_t> train_indexes;
    validation_indexes.clear();
    train_indexes.reserve(y_size);
    validation_indexes.reserve(y_size);
    for (Eigen::Index i = 0; i < cv_observations_in_fold.rows(); ++i)
    {
        bool training_observation{cv_observations_in_fold[i] == 1};
        bool validation_observation{cv_observations_in_fold[i] == -1};
        if (training_observation)
            train_indexes.push_back(i);
        else if (validation_observation)
            validation_indexes.push_back(i);
    }
    train_indexes.shrink_to_fit();
    validation_indexes.shrink_to_fit();

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
    for (size_t i = 0; i < train_indexes.size(); ++i)
    {
        sample_weight_train[i] = sample_weight[train_indexes[i]];
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
    for (size_t i = 0; i < validation_indexes.size(); ++i)
    {
        sample_weight_validation[i] = sample_weight[validation_indexes[i]];
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

void APLRRegressor::scale_response_if_using_log_link_function()
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

void APLRRegressor::initialize(const std::vector<int> &monotonic_constraints)
{
    number_of_base_terms = static_cast<size_t>(X_train.cols());

    terms.clear();
    terms.reserve(m);

    terms_eligible_current.reserve(m);
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

    this->monotonic_constraints = monotonic_constraints;
    bool monotonic_constraints_provided{monotonic_constraints.size() > 0};
    if (monotonic_constraints_provided)
    {
        for (auto &term_eligible_current : terms_eligible_current)
        {
            term_eligible_current.set_monotonic_constraint(monotonic_constraints[term_eligible_current.base_term]);
        }
    }

    bool loss_function_is_group_mse_cycle{loss_function == "group_mse_cycle"};
    if (loss_function_is_group_mse_cycle)
    {
        setup_groups_for_group_mse_cycle();
        group_cycle_predictor_index = 0;
    }
    bool need_to_initialize_prediction_groups{(loss_function == "group_mse_cycle" && validation_tuning_metric == "default") ||
                                              validation_tuning_metric == "group_mse_by_prediction"};
    if (need_to_initialize_prediction_groups)
    {
        size_t max_groups{static_cast<size_t>(y_validation.rows())};
        size_t groups_used{std::min(group_mse_by_prediction_bins, max_groups)};
        for (size_t i = 0; i < groups_used; ++i)
        {
            unique_prediction_groups.insert(i);
        }
    }

    intercept = 0.0;
    intercept_steps = VectorXd::Constant(m, 0.0);
    linear_predictor_current = VectorXd::Constant(y_train.size(), 0.0);
    linear_predictor_null_model = VectorXd::Constant(y_train.size(), 0.0);
    linear_predictor_current_validation = VectorXd::Constant(y_validation.size(), 0.0);
    predictions_current = transform_linear_predictor_to_predictions(linear_predictor_current, link_function, calculate_custom_transform_linear_predictor_to_predictions_function);
    predictions_current_validation = transform_linear_predictor_to_predictions(linear_predictor_current_validation, link_function, calculate_custom_transform_linear_predictor_to_predictions_function);

    validation_error_steps.resize(m, 1);
    validation_error_steps.setConstant(std::numeric_limits<double>::infinity());

    update_gradient_and_errors();

    best_validation_error_so_far = std::numeric_limits<double>::infinity();
    best_m_so_far = 0;

    round_robin_update_of_existing_terms = false;

    ridge_penalty_weights.resize(X_train.cols());
    for (Eigen::Index i = 0; i < X_train.cols(); ++i)
    {
        ridge_penalty_weights[i] = (X_train.col(i).array() * sample_weight_train.array() * X_train.col(i).array()).sum();
    }
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

void APLRRegressor::setup_groups_for_group_mse_cycle()
{
    size_t max_groups{static_cast<size_t>(y_train.rows())};
    bool group_mse_cycle_min_obs_in_bin_is_too_high{group_mse_cycle_min_obs_in_bin > max_groups};
    if (group_mse_cycle_min_obs_in_bin_is_too_high)
        group_mse_cycle_min_obs_in_bin = max_groups;
    size_t groups_used{max_groups / group_mse_cycle_min_obs_in_bin};
    for (size_t i = 0; i < groups_used; ++i)
    {
        unique_groups_cycle_train.insert(i);
    }
    size_t cycles{static_cast<size_t>(X_train.cols())};
    group_cycle_train.reserve(cycles);
    for (Eigen::Index i = 0; i < X_train.cols(); ++i)
    {
        VectorXi group{create_groups_for_group_mse_sorted_by_vector(X_train.col(i), unique_groups_cycle_train)};
        group_cycle_train.push_back(group);
    }
}

VectorXi APLRRegressor::create_groups_for_group_mse_sorted_by_vector(const VectorXd &vector, const std::set<int> &unique_groups_in_vector)
{
    VectorXi group{VectorXi(vector.rows())};
    size_t observations_per_group{vector.size() / unique_groups_in_vector.size()};
    VectorXi sorted_prediction_index{sort_indexes_ascending(vector)};
    std::vector<int> unique_groups{unique_groups_in_vector.begin(), unique_groups_in_vector.end()};

    size_t current_group{0};
    size_t middle_observation{static_cast<size_t>(group.size()) / 2};
    for (size_t i = 0; i < middle_observation; ++i)
    {
        group[sorted_prediction_index[i]] = unique_groups[current_group];
        bool increment_group{(i + 1) % observations_per_group == 0};
        bool can_increment_group{current_group < unique_groups.size() - 1};
        if (increment_group && can_increment_group)
            ++current_group;
    }
    size_t minimum_group_in_next_step{current_group};
    current_group = unique_groups.size() - 1;
    for (size_t i = vector.size() - 1; i >= middle_observation; --i)
    {
        group[sorted_prediction_index[i]] = unique_groups[current_group];
        bool decrement_group{(vector.size() - i) % observations_per_group == 0};
        bool can_decrement_group{current_group > minimum_group_in_next_step};
        if (decrement_group && can_decrement_group)
            --current_group;
    }

    return group;
}

VectorXd APLRRegressor::calculate_neg_gradient_current()
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
        GroupData group_residuals_and_count{calculate_group_errors_and_count(y_train, predictions_current, group_train, unique_groups_train,
                                                                             sample_weight_train)};
        output = calculate_neg_gradient_current_for_group_mse(group_residuals_and_count, group_train, unique_groups_train);
    }
    else if (loss_function == "group_mse_cycle")
    {
        GroupData group_residuals_and_count{calculate_group_errors_and_count(y_train, predictions_current,
                                                                             group_cycle_train[group_cycle_predictor_index],
                                                                             unique_groups_cycle_train,
                                                                             sample_weight_train)};
        output = calculate_neg_gradient_current_for_group_mse(group_residuals_and_count, group_cycle_train[group_cycle_predictor_index],
                                                              unique_groups_cycle_train);
    }
    else if (loss_function == "mae")
    {
        double mae{calculate_mean_error(calculate_errors(y_train, predictions_current, sample_weight_train, "mae"), sample_weight_train)};
        output = (y_train.array() - predictions_current.array()).sign() * mae;
    }
    else if (loss_function == "quantile")
    {
        double mae{calculate_mean_error(calculate_errors(y_train, predictions_current, sample_weight_train, "mae"), sample_weight_train)};
        output = (y_train.array() - predictions_current.array()).sign() * mae;
        for (Eigen::Index i = 0; i < y_train.size(); ++i)
        {
            if (std::isless(y_train[i], predictions_current[i]))
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
    else if (loss_function == "huber")
    {
        double delta = dispersion_parameter;
        if (link_function == "log")
            delta *= scaling_factor_for_log_link_function;
        output = (y_train - predictions_current).array().max(-delta).min(delta);
    }
    else if (loss_function == "exponential_power")
    {
        double p = dispersion_parameter;
        ArrayXd residuals = y_train.array() - predictions_current.array();
        output = p * residuals.abs().pow(p - 1.0) * residuals.sign();
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

    if (faster_convergence && (link_function == "identity" || link_function == "log"))
    {
        double standard_deviation_of_neg_gradient{calculate_standard_deviation(output, sample_weight_train)};
        if (is_approximately_zero(standard_deviation_of_neg_gradient))
        {
            return output;
        }

        ArrayXd denominator{ArrayXd::Ones(y_train.size())};
        if (link_function != "identity")
        {
            denominator = differentiate_predictions_wrt_linear_predictor().array();
        }

        double desired_standard_deviation{
            calculate_standard_deviation((y_train - predictions_current).array() / denominator, sample_weight_train)};
        double adjustment_factor = desired_standard_deviation / standard_deviation_of_neg_gradient;

        if (std::isfinite(adjustment_factor))
            output *= adjustment_factor;
    }

    return output;
}

VectorXd APLRRegressor::calculate_neg_gradient_current_for_group_mse(GroupData &group_residuals_and_count, const VectorXi &group,
                                                                     const std::set<int> &unique_groups)
{
    VectorXd output{VectorXd(y_train.rows())};
    for (Eigen::Index i = 0; i < y_train.size(); ++i)
    {
        output[i] = group_residuals_and_count.error[group[i]];
    }

    return output;
}

VectorXd APLRRegressor::differentiate_predictions_wrt_linear_predictor()
{
    if (link_function == "logit")
        return 10.0 / 4.0 * (linear_predictor_current.array() / 2.0).cosh().array().pow(-2);
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

void APLRRegressor::execute_boosting_steps(Eigen::Index fold_index)
{
    stopped_early = false;
    abort_boosting = false;
    for (size_t boosting_step = 0; boosting_step < m; ++boosting_step)
    {
        linear_effects_only_in_this_boosting_step = num_first_steps_with_linear_effects_only > boosting_step;
        non_linear_effects_allowed_in_this_boosting_step = boosting_steps_before_interactions_are_allowed > boosting_step && !linear_effects_only_in_this_boosting_step;
        bool last_linear_effects_only_step{linear_effects_only_in_this_boosting_step && boosting_step == num_first_steps_with_linear_effects_only - 1};
        bool last_step_before_interactions{non_linear_effects_allowed_in_this_boosting_step && boosting_step == boosting_steps_before_interactions_are_allowed - 1};
        execute_boosting_step(boosting_step, fold_index);
        if (stopped_early)
        {
            if (linear_effects_only_in_this_boosting_step)
                boosting_step = std::min(num_first_steps_with_linear_effects_only - 1, m - 1);
            else if (non_linear_effects_allowed_in_this_boosting_step)
                boosting_step = std::min(boosting_steps_before_interactions_are_allowed - 1, m - 1);
            best_m_so_far = boosting_step;
            stopped_early = false;
        }
        else if ((last_linear_effects_only_step || last_step_before_interactions) && boosting_step + 1 < m)
            find_optimal_m_and_update_model_accordingly();
        if (abort_boosting)
            break;
        if (loss_function == "group_mse_cycle")
        {
            bool predictor_index_cannot_increase{group_cycle_predictor_index >= group_cycle_train.size() - 1};
            if (predictor_index_cannot_increase)
                group_cycle_predictor_index = 0;
            else
                ++group_cycle_predictor_index;
        }
    }
}

void APLRRegressor::execute_boosting_step(size_t boosting_step, Eigen::Index fold_index)
{
    if (!round_robin_update_of_existing_terms)
    {
        model_has_changed_in_this_boosting_step = false;
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
                    prepare_for_round_robin_coefficient_updates_if_max_terms_has_been_reached();
                    if (round_robin_update_of_existing_terms)
                        break;
                }
            }
        }
        if (!abort_boosting && !round_robin_update_of_existing_terms)
        {
            std::vector<size_t> term_indexes{create_term_indexes(terms_eligible_current)};
            estimate_split_point_for_each_term(terms_eligible_current, term_indexes);
            best_term_index = find_best_term_index(terms_eligible_current, term_indexes);
            consider_interactions(predictor_indexes, boosting_step);
            select_the_best_term_and_update_errors(boosting_step);
            prepare_for_round_robin_coefficient_updates_if_max_terms_has_been_reached();
        }
        update_coefficient_steps(boosting_step);
        if (!model_has_changed_in_this_boosting_step)
        {
            if (linear_effects_only_in_this_boosting_step || non_linear_effects_allowed_in_this_boosting_step)
            {
                find_optimal_m_and_update_model_accordingly();
                stopped_early = true;
            }
            else
            {
                abort_boosting = true;
                if (verbosity >= 1)
                {
                    std::cout << "No further reduction in training loss was possible. Terminating the boosting procedure.\n";
                }
            }
        }
        abort_boosting_when_no_validation_error_improvement_in_the_last_early_stopping_rounds(boosting_step);
        if (abort_boosting)
            return;
        if (!round_robin_update_of_existing_terms)
            update_term_eligibility();
    }
    else
        update_a_term_coefficient_round_robin(boosting_step);
    print_summary_after_boosting_step(boosting_step, fold_index);
}

void APLRRegressor::update_intercept(size_t boosting_step)
{
    double intercept_update;
    intercept_update = v * (neg_gradient_current.array() * sample_weight_train.array()).sum() / sample_weight_train.array().sum();
    if (model_has_changed_in_this_boosting_step == false)
        model_has_changed_in_this_boosting_step = !is_approximately_equal(intercept_update, 0.0);
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

void APLRRegressor::prepare_for_round_robin_coefficient_updates_if_max_terms_has_been_reached()
{
    if (!round_robin_update_of_existing_terms)
    {
        max_terms_reached = max_terms > 0 && terms.size() >= max_terms;
        if (max_terms_reached)
        {
            number_of_eligible_terms = 1;
            round_robin_update_of_existing_terms = true;
            terms_eligible_current = terms;
            term_to_update_in_this_boosting_step = 0;
        }
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
    neg_gradient_current = calculate_neg_gradient_current();
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
    bool multithreading{cores_to_use > 1 && terms_indexes.size() > 1};

    if (multithreading)
    {
        std::vector<std::future<void>> results;
        for (size_t i = 0; i < terms_indexes.size(); ++i)
        {
            results.emplace_back(
                thread_pool->enqueue([&terms, &terms_indexes, i, this]
                                     { terms[terms_indexes[i]].estimate_split_point(
                                           this->X_train, this->neg_gradient_current, this->sample_weight_train, this->bins,
                                           this->predictor_learning_rates[terms[terms_indexes[i]].base_term],
                                           this->predictor_min_observations_in_split[terms[terms_indexes[i]].base_term],
                                           this->linear_effects_only_in_this_boosting_step,
                                           this->predictor_penalties_for_non_linearity[terms[terms_indexes[i]].base_term],
                                           this->predictor_penalties_for_interactions[terms[terms_indexes[i]].base_term],
                                           this->ridge_penalty,
                                           this->ridge_penalty_weights[terms[terms_indexes[i]].base_term]); }));
        }
        for (auto &&result : results)
        {
            result.get();
        }
    }
    else
    {
        for (size_t i = 0; i < terms_indexes.size(); ++i)
        {
            terms[terms_indexes[i]].estimate_split_point(X_train, neg_gradient_current, sample_weight_train, bins,
                                                         predictor_learning_rates[terms[terms_indexes[i]].base_term],
                                                         predictor_min_observations_in_split[terms[terms_indexes[i]].base_term],
                                                         linear_effects_only_in_this_boosting_step,
                                                         predictor_penalties_for_non_linearity[terms[terms_indexes[i]].base_term],
                                                         predictor_penalties_for_interactions[terms[terms_indexes[i]].base_term],
                                                         ridge_penalty,
                                                         ridge_penalty_weights[terms[terms_indexes[i]].base_term]);
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
    bool consider_interactions{terms.size() > 0 && max_interaction_level > 0 && interactions_eligible < max_interactions && boosting_step >= boosting_steps_before_interactions_are_allowed && std::isless(penalty_for_interactions, 1.0)};
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
                bool model_term_without_given_terms_can_be_a_given_term{
                    model_term_without_given_terms.get_monotonic_constraint() == 0 || monotonic_constraints_ignore_interactions == true};
                if (model_term_without_given_terms_can_be_a_given_term)
                    model_term_with_added_given_term.given_terms.push_back(model_term_without_given_terms);
                add_necessary_given_terms_to_interaction(interaction, model_term_with_added_given_term);
                bool interaction_only_uses_one_base_term{interaction.term_uses_just_these_predictors({interaction.base_term})};
                if (interaction_only_uses_one_base_term)
                    continue;
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
                if (interactions_to_consider[sorted_indexes_of_errors_for_interactions_to_consider[j]].get_interaction_level() > 0)
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

    if (model_has_changed_in_this_boosting_step == false)
        model_has_changed_in_this_boosting_step = !is_approximately_equal(terms_eligible_current[best_term_index].coefficient, 0.0);
    linear_predictor_update = terms_eligible_current[best_term_index].calculate_contribution_to_linear_predictor(X_train);
    linear_predictor_update_validation = terms_eligible_current[best_term_index].calculate_contribution_to_linear_predictor(X_validation);
    update_linear_predictor_and_predictions();
    update_gradient_and_errors();
    double backup_of_validation_error{validation_error_steps.col(0)[boosting_step]};
    calculate_and_validate_validation_error(boosting_step);
    if (abort_boosting)
        validation_error_steps.col(0)[boosting_step] = backup_of_validation_error;
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
    terms.push_back(Term(terms_eligible_current[best_term_index]));
}

void APLRRegressor::update_coefficient_steps(size_t boosting_step)
{
    for (auto &term : terms)
    {
        term.coefficient_steps[boosting_step] = term.coefficient;
    }
}

double APLRRegressor::calculate_quantile_mean_response(const VectorXd &predictions, bool top_quantile)
{
    double quantile_value{calculate_quantile(predictions, quantile, sample_weight_validation)};

    VectorXd predictions_in_quantile;
    if (top_quantile)
    {
        predictions_in_quantile = (predictions.array() >= quantile_value).cast<double>();
    }
    else
    {
        predictions_in_quantile = (predictions.array() <= quantile_value).cast<double>();
    }

    VectorXd y_in_quantile{y_validation.array() * predictions_in_quantile.array()};
    VectorXd weights_in_quantile{sample_weight_validation.array() * predictions_in_quantile.array()};

    double mean_response{calculate_weighted_average(y_in_quantile, weights_in_quantile)};

    if (std::isnan(mean_response))
        return std::numeric_limits<double>::infinity();

    return mean_response;
}

void APLRRegressor::calculate_and_validate_validation_error(size_t boosting_step)
{
    validation_error_steps.col(0)[boosting_step] = calculate_validation_error(predictions_current_validation);
    bool validation_error_is_invalid{!std::isfinite(validation_error_steps.col(0)[boosting_step])};
    if (validation_error_is_invalid)
    {
        abort_boosting = true;
        std::string warning_message{"Warning: Encountered numerical problems when calculating validation error in the previous boosting step. Not continuing with further boosting steps. One potential reason is if the combination of loss_function and link_function is invalid. Another potential reason could be that too many observations have zero sample_weight."};
        std::cout << warning_message << "\n";
    }
}

double APLRRegressor::calculate_validation_error(const VectorXd &predictions, const std::string &metric)
{
    VectorXd predictions_used{predictions};
    if (link_function == "log")
    {
        predictions_used /= scaling_factor_for_log_link_function;
    }

    if (metric == "default")
    {
        if (loss_function == "custom_function")
        {
            try
            {
                return calculate_custom_loss_function(y_validation, predictions_used, sample_weight_validation, group_validation, other_data_validation);
            }
            catch (const std::exception &e)
            {
                std::string error_msg{"Error when calculating custom loss function: " + static_cast<std::string>(e.what())};
                throw std::runtime_error(error_msg);
            }
        }
        else if (loss_function == "group_mse_cycle")
        {
            return calculate_group_mse_by_prediction_validation_error(predictions_used);
        }
        else
            return calculate_mean_error(calculate_errors(y_validation, predictions_used, sample_weight_validation, loss_function, dispersion_parameter, group_validation, unique_groups_validation, quantile), sample_weight_validation);
    }
    else if (metric == "mse")
        return calculate_mean_error(calculate_errors(y_validation, predictions_used, sample_weight_validation, MSE_LOSS_FUNCTION), sample_weight_validation);
    else if (metric == "mae")
        return calculate_mean_error(calculate_errors(y_validation, predictions_used, sample_weight_validation, "mae"), sample_weight_validation);
    else if (metric == "negative_gini")
        return -calculate_gini(y_validation, predictions_used, sample_weight_validation) / calculate_gini(y_validation, y_validation, sample_weight_validation);
    else if (metric == "group_mse")
    {
        bool group_is_not_provided{group_validation.rows() == 0};
        if (group_is_not_provided)
            throw std::runtime_error("When validation_tuning_metric is group_mse then the group argument in fit() must be provided.");
        return calculate_mean_error(calculate_errors(y_validation, predictions_used, sample_weight_validation, "group_mse", dispersion_parameter, group_validation, unique_groups_validation, quantile), sample_weight_validation);
    }
    else if (metric == "huber")
        return calculate_mean_error(calculate_errors(y_validation, predictions_used, sample_weight_validation, "huber", dispersion_parameter), sample_weight_validation);
    else if (metric == "group_mse_by_prediction")
    {
        return calculate_group_mse_by_prediction_validation_error(predictions_used);
    }
    else if (metric == "custom_function")
    {
        try
        {
            return calculate_custom_validation_error_function(y_validation, predictions_used, sample_weight_validation, group_validation, other_data_validation);
        }
        catch (const std::exception &e)
        {
            std::string error_msg{"Error when calculating custom validation error function: " + static_cast<std::string>(e.what())};
            throw std::runtime_error(error_msg);
        }
    }
    else if (metric == "neg_top_quantile_mean_response")
    {
        double mean_response{calculate_quantile_mean_response(predictions_used, true)};
        if (std::isinf(mean_response))
        {
            return mean_response;
        }
        return -mean_response;
    }
    else if (metric == "bottom_quantile_mean_response")
    {
        return calculate_quantile_mean_response(predictions_used, false);
    }
    else
        throw std::runtime_error(metric + " is an invalid validation_tuning_metric.");
}

double APLRRegressor::calculate_group_mse_by_prediction_validation_error(const VectorXd &predictions)
{
    VectorXi group{create_groups_for_group_mse_sorted_by_vector(predictions, unique_prediction_groups)};
    return calculate_mean_error(calculate_errors(y_validation, predictions, sample_weight_validation, "group_mse_cycle",
                                                 dispersion_parameter, group, unique_prediction_groups, quantile),
                                sample_weight_validation);
}

void APLRRegressor::update_term_eligibility()
{
    bool eligibility_is_used{ineligible_boosting_steps_added > 0 && max_eligible_terms > 0};
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

void APLRRegressor::update_a_term_coefficient_round_robin(size_t boosting_step)
{
    update_intercept(boosting_step);
    terms_eligible_current[term_to_update_in_this_boosting_step].estimate_split_point(X_train, neg_gradient_current, sample_weight_train,
                                                                                      bins,
                                                                                      predictor_learning_rates[terms_eligible_current[term_to_update_in_this_boosting_step].base_term],
                                                                                      predictor_min_observations_in_split[terms_eligible_current[term_to_update_in_this_boosting_step].base_term],
                                                                                      linear_effects_only_in_this_boosting_step,
                                                                                      predictor_penalties_for_non_linearity[terms_eligible_current[term_to_update_in_this_boosting_step].base_term],
                                                                                      predictor_penalties_for_interactions[terms_eligible_current[term_to_update_in_this_boosting_step].base_term],
                                                                                      ridge_penalty,
                                                                                      ridge_penalty_weights[terms_eligible_current[term_to_update_in_this_boosting_step].base_term],
                                                                                      true);
    terms[term_to_update_in_this_boosting_step].coefficient += terms_eligible_current[term_to_update_in_this_boosting_step].coefficient;
    linear_predictor_update = terms_eligible_current[term_to_update_in_this_boosting_step].calculate_contribution_to_linear_predictor(X_train);
    linear_predictor_update_validation = terms_eligible_current[term_to_update_in_this_boosting_step].calculate_contribution_to_linear_predictor(X_validation);
    update_linear_predictor_and_predictions();
    update_gradient_and_errors();
    calculate_and_validate_validation_error(boosting_step);
    update_coefficient_steps(boosting_step);
    abort_boosting_when_no_validation_error_improvement_in_the_last_early_stopping_rounds(boosting_step);
    if (abort_boosting)
        return;
    ++term_to_update_in_this_boosting_step;
    bool term_to_update_in_next_boosting_step_must_be_reset_to_zero{term_to_update_in_this_boosting_step >= terms.size()};
    if (term_to_update_in_next_boosting_step_must_be_reset_to_zero)
        term_to_update_in_this_boosting_step = 0;
}

void APLRRegressor::print_summary_after_boosting_step(size_t boosting_step, Eigen::Index fold_index)
{
    if (verbosity >= 2)
    {
        std::cout << "Fold: " << fold_index << ". Boosting step: " << boosting_step + 1 << ". Model terms: " << terms.size() << ". Terms eligible: " << number_of_eligible_terms << ". Validation error: " << validation_error_steps.col(0)[boosting_step] << ".\n";
    }
}

void APLRRegressor::abort_boosting_when_no_validation_error_improvement_in_the_last_early_stopping_rounds(size_t boosting_step)
{
    bool validation_error_is_better{std::isless(validation_error_steps.col(0)[boosting_step], best_validation_error_so_far)};
    if (validation_error_is_better)
    {
        best_validation_error_so_far = validation_error_steps.col(0)[boosting_step];
        best_m_so_far = boosting_step;
    }
    else
    {
        bool no_improvement_for_too_long{boosting_step > best_m_so_far + early_stopping_rounds};
        if (no_improvement_for_too_long)
        {
            if (linear_effects_only_in_this_boosting_step || non_linear_effects_allowed_in_this_boosting_step)
            {
                find_optimal_m_and_update_model_accordingly();
                stopped_early = true;
            }
            else
            {
                abort_boosting = true;
                if (verbosity >= 1)
                    std::cout << "Aborting boosting because of no validation error improvement in the last " << std::to_string(early_stopping_rounds) << " steps.\n";
            }
        }
    }
}

void APLRRegressor::print_final_summary()
{
    if (verbosity >= 1)
    {
        std::cout << "Model terms: " << terms.size() << ". Terms available in final boosting step: " << terms_eligible_current.size() << ".\n";
    }
}

void APLRRegressor::find_optimal_m_and_update_model_accordingly()
{
    size_t best_boosting_step_index;
    validation_error_steps.col(0).minCoeff(&best_boosting_step_index);
    intercept = intercept_steps[best_boosting_step_index];
    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].coefficient = terms[i].coefficient_steps[best_boosting_step_index];
    }
    m_optimal = best_boosting_step_index + 1;
}

void APLRRegressor::merge_similar_terms(const MatrixXd &X)
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
                    VectorXd values_i{terms[i].calculate(X)};
                    VectorXd values_j{terms[j].calculate(X)};
                    bool terms_are_similar{all_are_equal(values_i, values_j)};
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

void APLRRegressor::correct_mean_bias()
{
    if (link_function == "identity" || link_function == "log")
    {
        VectorXd predictions_train{predict(X_train, false)};
        double mean_y = calculate_weighted_average(y_train, sample_weight_train);
        double mean_pred = calculate_weighted_average(predictions_train, sample_weight_train);

        double bias_adjustment = 0.0;
        if (link_function == "identity")
        {
            bias_adjustment = mean_y - mean_pred;
        }
        else if (link_function == "log")
        {
            if (mean_pred > 0 && mean_y > 0)
                bias_adjustment = std::log(mean_y / mean_pred);
        }

        intercept += bias_adjustment;
    }
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
        scaling_factor_for_log_link_function = 1.0;
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
        set_term_affiliations(temp);
    }
    else
    {
        set_term_names(X_names);
        set_term_affiliations(X_names);
    }
}

void APLRRegressor::set_term_coefficients()
{
    term_coefficients.resize(terms.size() + 1);
    term_coefficients[0] = intercept;
    for (size_t i = 0; i < terms.size(); ++i)
    {
        term_coefficients[i + 1] = terms[i].coefficient;
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
    }

    term_names.resize(terms.size() + 1);
    term_names[0] = "Intercept";
    for (size_t i = 0; i < terms.size(); ++i)
    {
        term_names[i + 1] = terms[i].name;
    }
}

void APLRRegressor::set_term_affiliations(const std::vector<std::string> &X_names)
{
    for (auto &term : terms)
    {
        std::vector<size_t> base_terms_used_in_term{term.get_unique_base_terms_used_in_this_term()};
        for (size_t i = 0; i < base_terms_used_in_term.size(); ++i)
        {
            if (i == 0)
                term.predictor_affiliation = X_names[base_terms_used_in_term[i]];
            else
                term.predictor_affiliation = term.predictor_affiliation + " & " + X_names[base_terms_used_in_term[i]];
        }
    }

    term_affiliations.resize(terms.size());
    for (size_t i = 0; i < terms.size(); ++i)
    {
        term_affiliations[i] = terms[i].predictor_affiliation;
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
        if (std::isless(temp_split_point, 0.0))
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

VectorXd APLRRegressor::calculate_feature_importance(const MatrixXd &X, const VectorXd &sample_weight)
{
    validate_that_model_can_be_used(X);
    validate_sample_weight(X, sample_weight);
    VectorXd feature_importance{VectorXd::Zero(number_of_unique_term_affiliations)};
    for (size_t i = 0; i < number_of_unique_term_affiliations; ++i)
    {
        VectorXd contribution{VectorXd::Zero(X.rows())};
        for (auto &term : terms)
        {
            bool term_belongs_to_affiliation{unique_term_affiliation_map[term.predictor_affiliation] == i};
            if (term_belongs_to_affiliation)
                contribution += term.calculate_contribution_to_linear_predictor(X);
        }
        feature_importance[i] = calculate_standard_deviation(contribution, sample_weight);
    }
    return feature_importance;
}

void APLRRegressor::validate_sample_weight(const MatrixXd &X, const VectorXd &sample_weight)
{
    bool sample_weight_is_provided{sample_weight.size() > 0};
    if (sample_weight_is_provided)
    {
        bool sample_weight_is_invalid{sample_weight.rows() != X.rows()};
        if (sample_weight_is_invalid)
            throw std::runtime_error("If sample_weight is provided then it needs to contain as many rows as X does.");
    }
}

VectorXd APLRRegressor::calculate_term_importance(const MatrixXd &X, const VectorXd &sample_weight)
{
    validate_that_model_can_be_used(X);
    validate_sample_weight(X, sample_weight);
    VectorXd term_importance = VectorXd::Constant(terms.size(), 0);
    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd contrib{terms[i].calculate_contribution_to_linear_predictor(X)};
        double std_dev_of_contribution(calculate_standard_deviation(contrib, sample_weight));
        term_importance[i] = std_dev_of_contribution;
    }
    return term_importance;
}

MatrixXd APLRRegressor::calculate_local_feature_contribution(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    MatrixXd output{MatrixXd::Constant(X.rows(), number_of_unique_term_affiliations, 0)};

    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd contrib{terms[i].calculate_contribution_to_linear_predictor(X)};
        size_t column{unique_term_affiliation_map[terms[i].predictor_affiliation]};
        output.col(column) += contrib;
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

void APLRRegressor::write_output_to_cv_fold_models(Eigen::Index fold_index)
{
    cv_fold_models[fold_index].intercept = intercept;
    cv_fold_models[fold_index].terms = terms;
    cv_fold_models[fold_index].validation_error_steps = validation_error_steps;
    cv_fold_models[fold_index].validation_error = min_validation_error_for_current_fold;
    cv_fold_models[fold_index].m_optimal = get_optimal_m();
    cv_fold_models[fold_index].fold_index = fold_index;
    cv_fold_models[fold_index].min_training_prediction_or_response = min_training_prediction_or_response;
    cv_fold_models[fold_index].max_training_prediction_or_response = max_training_prediction_or_response;
    cv_fold_models[fold_index].sample_weight_train_sum = sample_weight_train.sum();
}

void APLRRegressor::cleanup_after_fit()
{
    terms.shrink_to_fit();
    X_train.resize(0, 0);
    y_train.resize(0);
    validation_indexes.clear();
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
    monotonic_constraints.clear();
    group_train.resize(0);
    group_validation.resize(0);
    unique_groups_train.clear();
    unique_groups_validation.clear();
    interactions_eligible = 0;
    other_data_train.resize(0, 0);
    other_data_validation.resize(0, 0);
    unique_prediction_groups.clear();
    group_cycle_train.clear();
    unique_groups_cycle_train.clear();
    ridge_penalty_weights.resize(0);
}

void APLRRegressor::check_term_integrity()
{
    for (auto &term : terms)
    {
        for (auto &given_term : term.given_terms)
        {
            bool same_base_term{term.base_term == given_term.base_term};
            if (same_base_term)
            {
                bool given_term_has_no_split_point{!std::isfinite(given_term.split_point)};
                bool given_term_has_the_same_direction_right{term.direction_right == given_term.direction_right};
                bool given_term_has_incorrect_split_point;
                if (term.direction_right)
                {
                    given_term_has_incorrect_split_point = std::islessequal(given_term.split_point, term.split_point);
                }
                else
                {
                    given_term_has_incorrect_split_point = std::isgreaterequal(given_term.split_point, term.split_point);
                }
                if (given_term_has_no_split_point)
                    throw std::runtime_error("Bug: Interaction in term " + term.name + " has no split point.");
                if (given_term_has_the_same_direction_right)
                    throw std::runtime_error("Bug: Interaction in term " + term.name + " has an incorrect direction_right.");
                if (given_term_has_incorrect_split_point)
                    throw std::runtime_error("Bug: Interaction in term " + term.name + " has an incorrect split_point.");
            }
        }
    }
}

void APLRRegressor::create_final_model(const MatrixXd &X, const VectorXd &sample_weight)
{
    compute_fold_weights();
    update_intercept_and_term_weights();
    create_terms(X);
    estimate_term_importances(X, sample_weight);
    sort_terms();
    calculate_other_term_vectors();
    compute_cv_error();
    concatenate_validation_error_steps();
    find_final_min_and_max_training_predictions_or_responses();
    compute_max_optimal_m();
    correct_term_names_coefficients_and_affiliations();
    feature_importance = calculate_feature_importance(X, sample_weight);

    cleanup_after_fit();
    additional_cleanup_after_creating_final_model();
}

void APLRRegressor::compute_fold_weights()
{
    double sum_training_weights{0.0};
    for (auto &cv_fold_model : cv_fold_models)
    {
        sum_training_weights += cv_fold_model.sample_weight_train_sum;
    }
    for (auto &cv_fold_model : cv_fold_models)
    {
        cv_fold_model.fold_weight = cv_fold_model.sample_weight_train_sum / sum_training_weights;
    }
}

void APLRRegressor::update_intercept_and_term_weights()
{
    for (auto &cv_fold_model : cv_fold_models)
    {
        cv_fold_model.intercept *= cv_fold_model.fold_weight;
        for (auto &term : cv_fold_model.terms)
        {
            term.coefficient *= cv_fold_model.fold_weight;
        }
    }
}

void APLRRegressor::create_terms(const MatrixXd &X)
{
    intercept = 0.0;
    terms.clear();
    for (auto &cv_fold_model : cv_fold_models)
    {
        intercept += cv_fold_model.intercept;
        terms.insert(terms.end(), cv_fold_model.terms.begin(), cv_fold_model.terms.end());
    }
    merge_similar_terms(X);
    remove_unused_terms();
}

void APLRRegressor::estimate_term_importances(const MatrixXd &X, const VectorXd &sample_weight)
{
    term_importance = calculate_term_importance(X, sample_weight);
    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].estimated_term_importance = term_importance[i];
    }
}

void APLRRegressor::sort_terms()
{
    std::sort(terms.begin(), terms.end(),
              [](const Term &a, const Term &b)
              { return std::isgreater(a.estimated_term_importance, b.estimated_term_importance) ||
                       (is_approximately_equal(a.estimated_term_importance, b.estimated_term_importance) && (a.base_term < b.base_term)) ||
                       (is_approximately_equal(a.estimated_term_importance, b.estimated_term_importance) && (a.base_term == b.base_term) &&
                        std::isless(a.coefficient, b.coefficient)); });

    for (size_t i = 0; i < terms.size(); ++i)
    {
        term_importance[i] = terms[i].estimated_term_importance;
    }
}

void APLRRegressor::calculate_other_term_vectors()
{
    term_main_predictor_indexes = VectorXi(terms.size());
    term_interaction_levels = VectorXi(terms.size());
    for (size_t i = 0; i < terms.size(); ++i)
    {
        term_main_predictor_indexes[i] = terms[i].base_term;
        term_interaction_levels[i] = terms[i].get_interaction_level();
    }
}

void APLRRegressor::compute_cv_error()
{
    cv_error = 0.0;
    for (auto &cv_fold_model : cv_fold_models)
    {
        cv_error += cv_fold_model.validation_error * cv_fold_model.fold_weight;
    }
}

void APLRRegressor::concatenate_validation_error_steps()
{
    validation_error_steps = MatrixXd(validation_error_steps.rows(), cv_fold_models.size());
    for (auto &cv_fold_model : cv_fold_models)
    {
        validation_error_steps.col(cv_fold_model.fold_index) = cv_fold_model.validation_error_steps;
    }
}

void APLRRegressor::find_final_min_and_max_training_predictions_or_responses()
{
    for (auto &cv_fold_model : cv_fold_models)
    {
        min_training_prediction_or_response = std::min(min_training_prediction_or_response, cv_fold_model.min_training_prediction_or_response);
        max_training_prediction_or_response = std::max(max_training_prediction_or_response, cv_fold_model.max_training_prediction_or_response);
    }
}

void APLRRegressor::compute_max_optimal_m()
{
    for (auto &cv_fold_model : cv_fold_models)
    {
        m_optimal = std::max(m_optimal, cv_fold_model.m_optimal);
    }
}

void APLRRegressor::correct_term_names_coefficients_and_affiliations()
{
    size_t terms_size_with_intercept{terms.size() + 1};
    term_names.resize(terms_size_with_intercept);
    term_coefficients.resize(terms_size_with_intercept);
    term_affiliations.resize(terms.size());

    term_names[0] = "Intercept";
    term_coefficients[0] = intercept;
    for (size_t i = 0; i < terms.size(); ++i)
    {
        term_names[i + 1] = terms[i].name;
        term_coefficients[i + 1] = terms[i].coefficient;
        term_affiliations[i] = terms[i].predictor_affiliation;
    }
    unique_term_affiliations = get_unique_strings_as_vector(term_affiliations);
    number_of_unique_term_affiliations = unique_term_affiliations.size();
    for (size_t i = 0; i < unique_term_affiliations.size(); ++i)
    {
        unique_term_affiliation_map[unique_term_affiliations[i]] = i;
    }
    base_predictors_in_each_unique_term_affiliation.resize(unique_term_affiliation_map.size());
    std::vector<std::set<size_t>> base_predictors_in_each_unique_term_affiliation_set(unique_term_affiliation_map.size());
    for (auto &term : terms)
    {
        std::vector<size_t> unique_base_terms_for_this_term{term.get_unique_base_terms_used_in_this_term()};
        base_predictors_in_each_unique_term_affiliation_set[unique_term_affiliation_map[term.predictor_affiliation]].insert(unique_base_terms_for_this_term.begin(), unique_base_terms_for_this_term.end());
    }
    for (size_t i = 0; i < base_predictors_in_each_unique_term_affiliation_set.size(); ++i)
    {
        base_predictors_in_each_unique_term_affiliation[i] = std::vector<size_t>(base_predictors_in_each_unique_term_affiliation_set[i].begin(), base_predictors_in_each_unique_term_affiliation_set[i].end());
    }
}

void APLRRegressor::additional_cleanup_after_creating_final_model()
{
    cv_fold_models.clear();
    intercept_steps.resize(0);
    for (auto &term : terms)
    {
        term.coefficient_steps.resize(0);
    }
    predictor_indexes.clear();
    prioritized_predictors_indexes.clear();
    interaction_constraints.clear();
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

MatrixXd APLRRegressor::calculate_local_term_contribution(const MatrixXd &X)
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

VectorXd APLRRegressor::calculate_local_contribution_from_selected_terms(const MatrixXd &X, const std::vector<size_t> &predictor_indexes)
{
    validate_that_model_can_be_used(X);

    VectorXd contribution_from_selected_terms{VectorXd::Constant(X.rows(), 0.0)};

    std::vector<size_t> term_indexes_used;
    term_indexes_used.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i)
    {
        if (terms[i].term_uses_just_these_predictors(predictor_indexes))
            term_indexes_used.push_back(i);
    }
    term_indexes_used.shrink_to_fit();

    for (auto &term_index_used : term_indexes_used)
    {
        contribution_from_selected_terms += terms[term_index_used].calculate_contribution_to_linear_predictor(X);
    }

    return contribution_from_selected_terms;
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

std::vector<std::string> APLRRegressor::get_term_affiliations()
{
    return term_affiliations;
}

std::vector<std::string> APLRRegressor::get_unique_term_affiliations()
{
    return unique_term_affiliations;
}

std::vector<std::vector<size_t>> APLRRegressor::get_base_predictors_in_each_unique_term_affiliation()
{
    return base_predictors_in_each_unique_term_affiliation;
}

VectorXd APLRRegressor::get_term_coefficients()
{
    return term_coefficients;
}

MatrixXd APLRRegressor::get_validation_error_steps()
{
    return validation_error_steps;
}

VectorXd APLRRegressor::get_feature_importance()
{
    return feature_importance;
}

VectorXd APLRRegressor::get_term_importance()
{
    return term_importance;
}

VectorXi APLRRegressor::get_term_main_predictor_indexes()
{
    return term_main_predictor_indexes;
}

VectorXi APLRRegressor::get_term_interaction_levels()
{
    return term_interaction_levels;
}

double APLRRegressor::get_intercept()
{
    return intercept;
}

size_t APLRRegressor::get_optimal_m()
{
    return m_optimal;
}

std::string APLRRegressor::get_validation_tuning_metric()
{
    return validation_tuning_metric;
}

std::map<double, double> APLRRegressor::get_main_effect_shape(size_t predictor_index)
{
    if (model_has_not_been_trained())
        throw std::runtime_error("The model must have been trained before using get_main_effect_shape().");

    std::map<double, double> main_effect_shape;

    std::string unique_term_affiliation{""};
    for (auto &term : terms)
    {
        if (term.term_uses_just_these_predictors({predictor_index}))
        {
            unique_term_affiliation = term.predictor_affiliation;
            break;
        }
    }
    bool no_term_uses_this_predictor{unique_term_affiliation == ""};
    if (no_term_uses_this_predictor)
        return main_effect_shape;

    std::vector<size_t> relevant_term_indexes{compute_relevant_term_indexes(unique_term_affiliation)};
    std::vector<double> split_points{compute_split_points(predictor_index, relevant_term_indexes)};

    MatrixXd X{MatrixXd::Constant(split_points.size(), 1, 0)};
    for (size_t i = 0; i < split_points.size(); ++i)
    {
        X(i, 0) = split_points[i];
    }
    VectorXd contribution_to_linear_predictor{compute_contribution_to_linear_predictor_from_specific_terms(X, relevant_term_indexes,
                                                                                                           {predictor_index})};
    for (size_t i = 0; i < split_points.size(); ++i)
    {
        main_effect_shape[split_points[i]] = contribution_to_linear_predictor[i];
    }

    return main_effect_shape;
}

std::vector<size_t> APLRRegressor::compute_relevant_term_indexes(const std::string &unique_term_affiliation)
{
    std::vector<size_t> relevant_term_indexes;
    relevant_term_indexes.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i)
    {
        size_t term_affiliation_index{unique_term_affiliation_map[unique_term_affiliation]};
        if (terms[i].term_uses_just_these_predictors(base_predictors_in_each_unique_term_affiliation[term_affiliation_index]))
            relevant_term_indexes.push_back(i);
    }
    relevant_term_indexes.shrink_to_fit();
    return relevant_term_indexes;
}

std::vector<double> APLRRegressor::compute_split_points(size_t predictor_index, const std::vector<size_t> &relevant_term_indexes)
{
    std::vector<double> split_points;
    size_t max_potential_split_points{(relevant_term_indexes.size() * 3 + 2) * 3};
    split_points.reserve(max_potential_split_points);
    for (auto &relevant_term_index : relevant_term_indexes)
    {
        bool split_point_exits{std::isfinite(terms[relevant_term_index].split_point)};
        bool base_term_is_appropriate{terms[relevant_term_index].base_term == predictor_index};
        if (split_point_exits && base_term_is_appropriate)
        {
            split_points.push_back(terms[relevant_term_index].split_point);
        }
        for (auto &given_term : terms[relevant_term_index].given_terms)
        {
            bool split_point_exits{std::isfinite(given_term.split_point)};
            bool base_term_is_appropriate{given_term.base_term == predictor_index};
            if (split_point_exits && base_term_is_appropriate)
            {
                split_points.push_back(given_term.split_point);
            }
        }
    }
    split_points.push_back(min_predictor_values_in_training[predictor_index]);
    split_points.push_back(max_predictor_values_in_training[predictor_index]);
    split_points = remove_duplicate_elements_from_vector(split_points);

    VectorXd split_point_increments{VectorXd(split_points.size() - 1)};
    for (Eigen::Index i = 0; i < split_point_increments.size(); ++i)
    {
        split_point_increments[i] = split_points[i + 1] - split_points[i];
    }
    double minimum_split_point_increment{split_point_increments.minCoeff()};
    double increment_around_split_points{minimum_split_point_increment / DIVISOR_IN_GET_MAIN_EFFECT_SHAPE_FUNCTION};

    size_t num_split_points_before_small_increments{split_points.size()};
    for (size_t i = 0; i < num_split_points_before_small_increments; ++i)
    {
        split_points.push_back(split_points[i] - increment_around_split_points);
        split_points.push_back(split_points[i] + increment_around_split_points);
    }
    split_points = remove_duplicate_elements_from_vector(split_points);
    split_points.shrink_to_fit();
    return split_points;
}

VectorXd APLRRegressor::compute_contribution_to_linear_predictor_from_specific_terms(const MatrixXd &X,
                                                                                     const std::vector<size_t> &term_indexes,
                                                                                     const std::vector<size_t> &base_predictors_used)
{
    VectorXd contribution_from_specific_terms = VectorXd::Zero(X.rows());
    std::unordered_map<size_t, size_t> X_map;
    for (size_t i = 0; i < base_predictors_used.size(); ++i)
    {
        X_map[base_predictors_used[i]] = i;
    }
    for (auto &term_index_used : term_indexes)
    {
        auto &term = terms[term_index_used];
        VectorXd contribution_from_this_term = term.coefficient * term.calculate_without_interactions(X.col(X_map[term.base_term]));
        for (auto &given_term : term.given_terms)
        {
            VectorXd values_from_given_term = given_term.calculate_without_interactions(X.col(X_map[given_term.base_term]));
            VectorXi indicator = calculate_indicator(values_from_given_term);
            contribution_from_this_term = contribution_from_this_term.array() * indicator.cast<double>().array();
        }
        contribution_from_specific_terms += contribution_from_this_term;
    }
    return contribution_from_specific_terms;
}

MatrixXd APLRRegressor::get_unique_term_affiliation_shape(const std::string &unique_term_affiliation, size_t max_rows_before_sampling, size_t additional_points)
{
    if (model_has_not_been_trained())
        throw std::runtime_error("The model must have been trained before using get_unique_term_affiliation_shape().");

    bool found{false};
    for (auto &affiliation : unique_term_affiliations)
    {
        if (affiliation == unique_term_affiliation)
        {
            found = true;
            break;
        }
    }
    if (!found)
        throw std::runtime_error("The unique term affiliation that was provided to get_unique_term_affiliation_shape() does not exist.");

    std::vector<size_t> relevant_term_indexes{compute_relevant_term_indexes(unique_term_affiliation)};
    size_t unique_term_affiliation_index{unique_term_affiliation_map[unique_term_affiliation]};
    size_t num_predictors_used_in_the_affiliation{base_predictors_in_each_unique_term_affiliation[unique_term_affiliation_index].size()};
    std::vector<std::vector<double>> split_points_in_each_predictor(num_predictors_used_in_the_affiliation);
    for (size_t i = 0; i < num_predictors_used_in_the_affiliation; ++i)
    {
        split_points_in_each_predictor[i] = compute_split_points(base_predictors_in_each_unique_term_affiliation[unique_term_affiliation_index][i], relevant_term_indexes);

        if (num_predictors_used_in_the_affiliation > 1 && additional_points > 0 && !split_points_in_each_predictor[i].empty())
        {
            double min_val = *std::min_element(split_points_in_each_predictor[i].begin(), split_points_in_each_predictor[i].end());
            double max_val = *std::max_element(split_points_in_each_predictor[i].begin(), split_points_in_each_predictor[i].end());
            std::vector<double> interpolated;
            interpolated.reserve(additional_points);
            for (size_t j = 1; j <= additional_points; ++j)
            {
                double val = min_val + (max_val - min_val) * j / (additional_points + 1);
                interpolated.push_back(val);
            }
            split_points_in_each_predictor[i].reserve(split_points_in_each_predictor[i].size() + additional_points);
            split_points_in_each_predictor[i].insert(split_points_in_each_predictor[i].end(), interpolated.begin(), interpolated.end());
            split_points_in_each_predictor[i] = remove_duplicate_elements_from_vector(split_points_in_each_predictor[i]);
        }
    }

    size_t num_split_point_combinations = 1;
    for (size_t i = 0; i < split_points_in_each_predictor.size(); ++i)
    {
        num_split_point_combinations *= split_points_in_each_predictor[i].size();
    }
    bool need_to_sample{num_split_point_combinations > max_rows_before_sampling};
    if (need_to_sample)
    {
        double num_split_point_combinations_sqrt = std::sqrt(static_cast<double>(num_split_point_combinations));
        double factor = std::pow(max_rows_before_sampling / num_split_point_combinations_sqrt, 1.0 / split_points_in_each_predictor.size());
        std::mt19937 seed(random_state);
        for (auto &split_points : split_points_in_each_predictor)
        {
            size_t current_num_observations = split_points.size();
            size_t num_observations_to_keep = std::round(factor * std::sqrt(current_num_observations));
            num_observations_to_keep = std::max<size_t>(1, num_observations_to_keep);
            if (current_num_observations > num_observations_to_keep)
            {
                std::shuffle(split_points.begin(), split_points.end(), seed);
                split_points.resize(num_observations_to_keep);
                std::sort(split_points.begin(), split_points.end());
            }
        }
    }

    MatrixXd output{generate_combinations_and_one_additional_column(split_points_in_each_predictor)};
    output.col(num_predictors_used_in_the_affiliation) = compute_contribution_to_linear_predictor_from_specific_terms(output.block(0, 0, output.rows(), output.cols() - 1),
                                                                                                                      relevant_term_indexes,
                                                                                                                      base_predictors_in_each_unique_term_affiliation[unique_term_affiliation_index]);

    return output;
}

double APLRRegressor::get_cv_error()
{
    return cv_error;
}

size_t APLRRegressor::get_num_cv_folds()
{
    return cv_y_all_folds.size();
}

void APLRRegressor::set_intercept(double value)
{
    if (model_has_not_been_trained())
        throw std::runtime_error("The model must be trained with fit() before set_intercept() can be run.");
    if (!std::isfinite(value))
        throw std::runtime_error("The new intercept must be finite.");
    intercept = value;
    term_coefficients[0] = value;
}

void APLRRegressor::remove_provided_custom_functions()
{
    calculate_custom_validation_error_function = {};
    calculate_custom_loss_function = {};
    calculate_custom_negative_gradient_function = {};
}

void APLRRegressor::validate_fold_index(size_t fold_index)
{
    if (get_num_cv_folds() == 0)
    {
        throw_cv_data_not_available_error("CV results");
    }
    if (fold_index >= get_num_cv_folds())
        throw std::runtime_error("fold_index is out of bounds.");
}

void APLRRegressor::throw_cv_data_not_available_error(const std::string &data_name)
{
    throw std::runtime_error(data_name + " are not available. This can happen if the model was trained with an older version of APLR or if clear_cv_results() has been called.");
}

VectorXd APLRRegressor::get_cv_validation_predictions(size_t fold_index)
{
    validate_fold_index(fold_index);
    if (cv_validation_predictions_all_folds[fold_index].size() == 0)
        throw_cv_data_not_available_error("CV validation predictions");
    return cv_validation_predictions_all_folds[fold_index];
}

VectorXd APLRRegressor::get_cv_y(size_t fold_index)
{
    validate_fold_index(fold_index);
    if (cv_y_all_folds[fold_index].size() == 0)
        throw_cv_data_not_available_error("CV y values");
    return cv_y_all_folds[fold_index];
}

VectorXd APLRRegressor::get_cv_sample_weight(size_t fold_index)
{
    validate_fold_index(fold_index);
    if (cv_sample_weight_all_folds[fold_index].size() == 0)
        throw_cv_data_not_available_error("CV sample weights");
    return cv_sample_weight_all_folds[fold_index];
}

VectorXi APLRRegressor::get_cv_validation_indexes(size_t fold_index)
{
    validate_fold_index(fold_index);
    if (cv_validation_indexes_all_folds[fold_index].size() == 0)
        throw_cv_data_not_available_error("CV validation indexes");
    return cv_validation_indexes_all_folds[fold_index];
}

void APLRRegressor::clear_cv_results()
{
    if (model_has_not_been_trained())
        throw std::runtime_error("The model must be trained with fit() before clear_cv_results() can be run.");
    cv_validation_predictions_all_folds.clear();
    cv_y_all_folds.clear();
    cv_sample_weight_all_folds.clear();
    cv_validation_indexes_all_folds.clear();
}